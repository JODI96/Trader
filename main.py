"""
main.py — Orderflow Scalping Bot Entry Point

Three modes:
  1. SIMULATION  — paper trading, no real orders
  2. ALERT       — signals only, manual entry required
  3. AUTO        — live Binance Futures trading

Usage:
  python main.py
  python main.py --mode sim
  python main.py --mode alert --symbol ETHUSDT
  python main.py --mode auto
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import queue
import signal as _signal
import sys
import time
import threading
from pathlib import Path
from typing import Optional

_LOCKFILE = Path("trader.pid")

def _is_process_alive(pid: int) -> bool:
    """Cross-platform check for whether a PID is still running."""
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_INFORMATION = 0x0400
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259   # STILL_ACTIVE
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _acquire_lock() -> None:
    """Prevent two bot instances running at once."""
    if _LOCKFILE.exists():
        try:
            pid = int(_LOCKFILE.read_text().strip())
            if _is_process_alive(pid):
                print(
                    f"\n[ERROR] Another bot instance is already running (PID {pid}).\n"
                    f"If that is wrong, delete '{_LOCKFILE}' and retry.\n"
                )
                sys.exit(1)
        except (ValueError, OSError):
            pass   # unreadable or stale lockfile — overwrite it
    _LOCKFILE.write_text(str(os.getpid()))
    atexit.register(_release_lock)

def _release_lock() -> None:
    try:
        _LOCKFILE.unlink(missing_ok=True)
    except Exception:
        pass

# ── Setup logging before any imports ────────────────────────────────────────
# Force UTF-8 on Windows console (fixes Δ, ✓, ░ etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

import config
from data.binance_client import BinanceClient
from data.economic_calendar import EconomicCalendar
from data.market_data import MarketDataStore, RawTrade, Signal
from indicators.atr import ATRIndicator
from indicators.cvd import CVDIndicator
from indicators.delta_volume import DeltaVolumeIndicator
from indicators.support_resistance import SupportResistance
from indicators.volume_profile import VolumeProfile
from indicators.vwap import VWAPIndicator
from modes.alert import AlertMode
from modes.auto_trade import AutoTradeMode
from modes.simulation import SimulationMode
from strategy.filters import TradeFilters
from strategy.risk_manager import RiskManager
from ui.dashboard import Dashboard
from ml.state_builder import build_state, compute_confluence, STATE_DIM
from ml.neural_agent import DQNAgent, HOLD, LONG, SHORT, ACTION_NAMES
from ml.param_tuner import ParameterTuner


# ──────────────────────────────────────────────────────────────────────────────
# Mode selection
# ──────────────────────────────────────────────────────────────────────────────

def choose_mode(arg_mode: Optional[str]) -> str:
    if arg_mode:
        return arg_mode.upper()

    from rich.console import Console
    from rich.prompt import Prompt
    con = Console()
    con.print("\n[bold bright_blue]Orderflow Scalping Bot[/bold bright_blue]\n")
    con.print("  [cyan]1[/cyan]  SIMULATION  — paper trading (no real money)")
    con.print("  [yellow]2[/yellow]  ALERT       — signal alerts only (manual entry)")
    con.print("  [red]3[/red]  AUTO        — live Binance Futures trading\n")

    choice = Prompt.ask("Select mode", choices=["1", "2", "3"], default="1")
    return {"1": "SIMULATION", "2": "ALERT", "3": "AUTO"}[choice]


# ──────────────────────────────────────────────────────────────────────────────
# Main bot
# ──────────────────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self, mode: str):
        self.mode      = mode
        self._running  = False
        self._sig_queue: queue.Queue[Signal] = queue.Queue()

        # ── Data layer ──────────────────────────────────────────────────────
        self.client   = BinanceClient()
        self.mds      = MarketDataStore()
        self.calendar = EconomicCalendar()

        # ── Indicators ──────────────────────────────────────────────────────
        self.atr  = ATRIndicator()
        self.dv   = DeltaVolumeIndicator()
        self.vwap = VWAPIndicator()
        self.cvd  = CVDIndicator()
        self.vp   = VolumeProfile()
        self.sr   = SupportResistance()

        # ── Strategy ────────────────────────────────────────────────────────
        self.risk_mgr = RiskManager(config.INITIAL_BALANCE,
                                    sim_mode=(mode in ("SIMULATION", "BACKTEST")))
        self.filters  = TradeFilters(self.atr, self.dv, self.risk_mgr, self.calendar,
                                     backtest_mode=(mode == "BACKTEST"))

        # ── Modes ───────────────────────────────────────────────────────────
        self.sim_mode   = SimulationMode(self.risk_mgr)
        self.alert_mode = AlertMode()

        self.auto_mode: Optional[AutoTradeMode] = None
        if mode == "AUTO":
            self.auto_mode = AutoTradeMode(self.client, self.risk_mgr)

        # ── Neural Network Agent ─────────────────────────────────────────────
        self._nn_agent = DQNAgent(state_dim=STATE_DIM, auto_mode=(mode == "AUTO"))
        # Pending NN experience for the currently open position
        self._pending_nn_state:  Optional[list]  = None
        self._pending_nn_action: Optional[int]   = None
        self._pending_nn_rr:     float           = 3.0   # default R:R for reward calc
        # Previous candle state — needed to record hold experiences
        self._prev_state:        Optional[list]  = None
        # Cooldown: candles remaining before next trade is allowed
        self._nn_cooldown:       int             = 0
        # Last scalp candle ts for chart trade markers
        self._last_chart_candle_ts: int          = 0

        # ── Parameter self-tuner ─────────────────────────────────────────────
        self._param_tuner = ParameterTuner()

        # ── Dashboard ───────────────────────────────────────────────────────
        self.dashboard = Dashboard(
            mode        = mode,
            risk_mgr    = self.risk_mgr,
            atr         = self.atr,
            dv          = self.dv,
            vwap        = self.vwap,
            cvd         = self.cvd,
            vp          = self.vp,
            sr          = self.sr,
            filters     = self.filters,
            calendar    = self.calendar,
            nn_agent    = self._nn_agent,
            param_tuner = self._param_tuner,
        )

        # ── Wire up callbacks ───────────────────────────────────────────────
        self.client.add_trade_callback(self.mds.feed_trade)
        self.mds.on_scalp(self._on_scalp_candle)
        self.mds.on_htf(self._on_htf_candle)

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(f"Starting bot in {self.mode} mode for {config.SYMBOL}")
        self._running = True

        # ── Warmup: fetch historical klines ─────────────────────────────────
        logger.info("Fetching historical data for warmup…")
        try:
            scalp_klines = self.client.get_klines(
                config.SYMBOL, "1m", limit=200
            )
            self.mds.seed_from_klines(scalp_klines, "scalp")

            htf_klines = self.client.get_klines(
                config.SYMBOL, "15m", limit=100
            )
            self.mds.seed_from_klines(htf_klines, "htf")

            # Feed historical candles through structural indicators only.
            # DV and CVD are NOT updated here because the seed data is 1-min
            # klines — their volume is ~4× a real 15s candle, which would
            # permanently inflate the volume baseline and block the volume filter.
            for c in list(self.mds.scalp_candles):
                self._warmup_indicators_scalp(c)
            for c in list(self.mds.htf_candles):
                self._update_indicators_htf(c)

            logger.info(
                f"Warmup complete: {len(self.mds.scalp_candles)} scalp, "
                f"{len(self.mds.htf_candles)} HTF candles"
            )
        except Exception as e:
            logger.warning(f"Warmup failed (starting cold): {e}")

        # ── Prefetch economic calendar ────────────────────────────────────
        threading.Thread(
            target=lambda: self.calendar.get_events(refresh=True),
            daemon=True,
        ).start()

        # ── Start WebSocket stream ────────────────────────────────────────
        self.client.start_stream(config.SYMBOL)
        logger.info("Live stream started")

        # ── Main loop with Rich dashboard ────────────────────────────────
        try:
            live = self.dashboard.start()
            with live:
                while self._running:
                    self._process_signals()
                    self._update_positions()
                    self.dashboard.update_price(self.mds.last_price)
                    self.dashboard.refresh()
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down…")
        self._running = False
        self.mds.flush()
        self.client.stop_stream()
        self._nn_agent.save()
        logger.info("Bot stopped.")

    def run_backtest(self, days: int) -> None:
        """
        High-speed historical replay for NN training.
        Downloads `days` days of 1m Binance Futures klines and replays them
        through the full indicator + NN + simulation pipeline.
        The WebSocket is never started; no Rich dashboard is shown.
        """
        from modes.backtest import BacktestRunner
        logger.info(f"Starting backtest: {days} days of {config.SYMBOL}")
        BacktestRunner(self).run(days)
        logger.info("Backtest complete.")

    # ── Candle handlers ──────────────────────────────────────────────────────

    def _on_scalp_candle(self, candle) -> None:
        self._update_indicators_scalp(candle)
        self.dashboard.chart.add_candle(candle)
        self._last_chart_candle_ts = candle.ts

        # ── Per-candle real-time log ─────────────────────────────────────
        from datetime import datetime
        ts       = datetime.fromtimestamp(candle.ts / 1000).strftime("%H:%M:%S")
        bub      = f" BUBBLE({self.dv.bubble_direction})" if self.dv.is_any_bubble else ""
        abs_flag = " ABSORB!" if self.cvd.absorption_detected else ""
        exh_flag = " EXHAUST!" if self.cvd.exhaustion_detected else ""
        ok, reason = self.filters.check_all(candle.ts)
        filt_str = "FILTERS:OK" if ok else f"BLOCKED:{reason}"
        logger.info(
            f"[{ts}] C:{candle.close:.2f}  "
            f"d:{candle.delta:+.1f}({self.dv.bubble_strength:.1f}x){bub}  "
            f"CVD:{self.cvd.cvd:+.0f}{abs_flag}{exh_flag}  "
            f"VWAP:{self.vwap.vwap:.2f}({self.vwap.zone})  "
            f"ATR:{self.atr.atr:.2f}  "
            f"{filt_str}"
        )

        # ── Build state vector from all indicators ───────────────────────
        current_state = build_state(self.vwap, self.cvd, self.dv,
                                    self.atr, self.vp, self.mds, self.sr, candle.ts)

        # ── NN decision ──────────────────────────────────────────────────
        nn_action = self._nn_agent.select_action(current_state)

        # Record deferred hold for the previous candle (when no trade opened).
        # Subsample 1/20: without this, ~14k HOLD records vs ~350 trades per cycle
        # floods the replay buffer with reward=0 experiences and the NN can't learn
        # to distinguish good trading states (Q-values collapse to ~0 everywhere).
        if self._prev_state is not None and self._pending_nn_state is None:
            if self._nn_agent._steps % 20 == 0:
                # Small positive reward when neither direction has good confluence —
                # teaches the NN that skipping low-quality setups has value.
                long_conf  = compute_confluence(current_state, "LONG")
                short_conf = compute_confluence(current_state, "SHORT")
                hold_reward = 0.04 if max(long_conf, short_conf) < 0.33 else 0.0
                self._nn_agent.record_hold(self._prev_state, current_state, hold_reward)
        self._prev_state = current_state

        # Log NN action when filters pass
        if ok:
            buf = len(self._nn_agent._buffer)
            buf_str = (f"buf={buf}/{self._nn_agent.BATCH_SIZE}(warmup)"
                       if buf < self._nn_agent.BATCH_SIZE else f"buf={buf}")
            logger.info(
                f"NN action: {ACTION_NAMES[nn_action]}  "
                f"e={self._nn_agent.epsilon:.3f}  {buf_str}"
            )

        # Tick down cooldown counter
        if self._nn_cooldown > 0:
            self._nn_cooldown -= 1

        # ── Trade entry: NN says LONG or SHORT, filters pass, no open position, cooldown expired
        if ok and nn_action != HOLD and self._pending_nn_state is None and self._nn_cooldown == 0:
            sig = self._build_nn_signal(nn_action, candle)
            if sig:
                # Snapshot the full sequence BEFORE perturbing params
                self._nn_agent.snapshot_open_seq()
                self._param_tuner.perturb()
                self._pending_nn_state  = current_state
                self._pending_nn_action = nn_action
                self._pending_nn_rr     = sig.rr_ratio
                self._prev_state        = None   # consumed by this trade
                self._sig_queue.put(sig)

    def _on_htf_candle(self, candle) -> None:
        self._update_indicators_htf(candle)

    def _build_nn_signal(self, nn_action: int, candle) -> Optional[Signal]:
        """
        Build a Signal from the NN decision using ATR-based SL/TP.
        SL distance = NN_SL_ATR_MULT × ATR from entry.
        TP distance = SL distance × MIN_RR_RATIO.
        Returns None if ATR not ready.
        """
        if not self.atr.is_ready or self.atr.atr <= 0:
            logger.debug("NN signal skipped: ATR not ready")
            return None

        price   = candle.close
        sl_dist = self.atr.atr * config.NN_SL_ATR_MULT
        tp_dist = sl_dist * config.MIN_RR_RATIO

        if nn_action == LONG:
            direction   = "LONG"
            stop_loss   = price - sl_dist
            take_profit = price + tp_dist
        else:
            direction   = "SHORT"
            stop_loss   = price + sl_dist
            take_profit = price - tp_dist

        valid_rr, rr = self.risk_mgr.check_rr(price, stop_loss, take_profit, direction)
        if not valid_rr:
            logger.debug(f"NN signal skipped: R:R {rr:.2f} < {config.MIN_RR_RATIO}")
            return None

        return Signal(
            direction   = direction,
            strategy    = "NN",
            entry_price = price,
            stop_loss   = stop_loss,
            take_profit = take_profit,
            confidence  = 3,
            rr_ratio    = rr,
            reason      = (
                f"NN:{ACTION_NAMES[nn_action]}  "
                f"ATR:{self.atr.atr:.2f}  "
                f"e={self._nn_agent.epsilon:.3f}"
            ),
            timestamp   = time.time(),
        )

    def _warmup_indicators_scalp(self, candle) -> None:
        """Warmup pass: structural indicators only — NOT volume-sensitive ones."""
        self.atr.update(candle)
        self.vwap.update(candle)
        self.vp.update(candle)
        self.sr.update(candle)

    def _update_indicators_scalp(self, candle) -> None:
        self.atr.update(candle)
        self.dv.update(candle)
        self.vwap.update(candle)
        self.cvd.update(candle)
        self.vp.update(candle)
        self.sr.update(candle)

    def _update_indicators_htf(self, candle) -> None:
        self.vwap.update_htf(candle)
        self.sr.update_htf(candle)

    # ── Signal dispatch ──────────────────────────────────────────────────────

    def _process_signals(self) -> None:
        while not self._sig_queue.empty():
            try:
                sig = self._sig_queue.get_nowait()
            except queue.Empty:
                break

            self.dashboard.log_signal(sig)
            logger.info(
                f">>> SIGNAL {sig.direction} | {sig.strategy} | "
                f"Entry:{sig.entry_price:.2f}  SL:{sig.stop_loss:.2f}  TP:{sig.take_profit:.2f}  "
                f"R:R 1:{sig.rr_ratio:.1f}  Conf:{sig.confidence}/3 | {sig.reason}"
            )

            if self.mode in ("SIMULATION", "BACKTEST"):
                pos = self.sim_mode.process_signal(sig)
                if pos:
                    logger.info(
                        f">>> POSITION OPENED [{pos.id}] {pos.direction} "
                        f"@ {pos.entry_price:.2f}  size:{pos.size:.4f} BTC  "
                        f"SL:{pos.stop_loss:.2f}  TP:{pos.take_profit:.2f}"
                    )
                    self.dashboard.open_position_str = (
                        f"{pos.direction} @ ${pos.entry_price:,.2f}  [{pos.strategy}]"
                    )
                    self.dashboard.chart_trade_open(
                        self._last_chart_candle_ts, pos.direction, pos.entry_price
                    )

            elif self.mode == "ALERT":
                self.alert_mode.process_signal(sig)

            elif self.mode == "AUTO" and self.auto_mode:
                pos = self.auto_mode.process_signal(sig)
                if pos:
                    self.dashboard.open_position_str = (
                        f"{pos.signal.direction} @ ${pos.entry_price:,.2f}  "
                        f"[{pos.signal.strategy}]"
                    )

    # ── Position management ──────────────────────────────────────────────────

    def _update_positions(self) -> None:
        price = self.mds.last_price
        if price == 0:
            return

        if self.mode == "SIMULATION":
            closed = self.sim_mode.update_price(price)
            if closed:
                emoji = "WIN +" if closed.pnl_usdt >= 0 else "LOSS "
                logger.info(
                    f">>> POSITION CLOSED [{closed.id}] {closed.status}  "
                    f"PnL: {'+' if closed.pnl_usdt >= 0 else ''}{closed.pnl_usdt:.2f} USDT  "
                    f"Exit:{closed.close_price:.2f}"
                )
                self.dashboard.open_position_str = "No open position"
                self.dashboard.log_close(closed.status, closed.pnl_usdt)
                self.dashboard.chart_trade_close(closed.status)
                self._record_nn_trade(closed.status, closed.close_price)
            self.dashboard.unrealized_pnl = self.sim_mode.get_open_unrealized_pnl()
            self.dashboard.sim_stats = self.sim_mode.stats()

        elif self.mode == "AUTO" and self.auto_mode:
            closed = self.auto_mode.update_price(price)
            if closed:
                self.dashboard.open_position_str = "No open position"
                self._record_nn_trade(closed.status, price)
            self.dashboard.unrealized_pnl = self.auto_mode.unrealized_pnl(price)
            self.dashboard.sim_stats = self.risk_mgr.summary()

        elif self.mode == "ALERT":
            self.dashboard.sim_stats = self.risk_mgr.summary()

    def _record_nn_trade(self, outcome: str, close_price: float, close_ts_ms: int = 0) -> None:
        """Feed closed trade outcome back to the NN agent and param tuner."""
        if self._pending_nn_state is None or self._pending_nn_action is None:
            return

        self.dashboard.chart_trade_close(outcome)

        # Base reward in R-multiples
        if outcome == "WIN":
            base_reward = self._pending_nn_rr
        elif outcome == "LOSS":
            base_reward = -1.0
        else:   # BE
            base_reward = -0.1

        direction  = "LONG" if self._pending_nn_action == LONG else "SHORT"
        confluence = compute_confluence(self._pending_nn_state, direction)

        if self.mode == "BACKTEST":
            # Plain R-multiple reward — 1m candles make confluence scores
            # unreliable, so confluence shaping adds noise to the gradient.
            shaped_reward = base_reward
        else:
            # Confluence-shaped reward for live modes (15s candles):
            #   WIN  high conf → bigger bonus  (×1.3)
            #   LOSS low  conf → bigger penalty (×1.3)
            if base_reward > 0:
                shaped_reward = base_reward * (0.7 + 0.6 * confluence)
            else:
                shaped_reward = base_reward * (1.3 - 0.6 * confluence)

        ts_ms = close_ts_ms if close_ts_ms > 0 else int(self.mds.last_trade_ts * 1000)
        close_state = build_state(self.vwap, self.cvd, self.dv,
                                  self.atr, self.vp, self.mds, self.sr, ts_ms)

        self._nn_agent.record_trade(
            self._pending_nn_state,
            self._pending_nn_action,
            shaped_reward,
            close_state,
        )

        # Param tuner: record shaped reward against the perturbation active
        # during this trade; triggers a parameter update every 30 trades
        self._param_tuner.record(shaped_reward)

        st = self._nn_agent.stats()
        loss_str = f"{st['last_loss']:.4f}" if st["last_loss"] is not None else "n/a"
        logger.info(
            f"NN trained: reward={shaped_reward:+.2f} "
            f"(base={base_reward:+.1f} conf={confluence:.2f})  "
            f"loss={loss_str}  e={st['epsilon']:.3f}  trades={st['trade_count']}"
        )

        # Reset pending experience and start cooldown
        self._pending_nn_state  = None
        self._pending_nn_action = None
        self._nn_cooldown       = config.NN_TRADE_COOLDOWN


# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orderflow Scalping Bot")
    p.add_argument("--mode", choices=["sim", "simulation", "alert", "auto", "backtest"],
                   help="Trading mode (default: interactive prompt)")
    p.add_argument("--symbol", default=None,
                   help="Override symbol (e.g. ETHUSDT)")
    p.add_argument("--days", type=int, default=None,
                   help="Days of history for backtest mode (default: BACKTEST_DAYS in config)")
    p.add_argument("--cycles", type=int, default=1,
                   help="How many times to repeat the backtest (default: 1)")
    return p.parse_args()


def main() -> None:
    _acquire_lock()
    args = parse_args()

    if args.symbol:
        config.SYMBOL = args.symbol.upper()

    arg_mode = None
    if args.mode in ("sim", "simulation"):
        arg_mode = "SIMULATION"
    elif args.mode == "alert":
        arg_mode = "ALERT"
    elif args.mode == "auto":
        arg_mode = "AUTO"
    elif args.mode == "backtest":
        arg_mode = "BACKTEST"

    # Backtest skips the interactive mode prompt and the WebSocket
    if arg_mode == "BACKTEST":
        days   = args.days if args.days is not None else config.BACKTEST_DAYS
        cycles = max(1, args.cycles)
        for cycle in range(1, cycles + 1):
            if cycles > 1:
                print(f"\n{'='*60}")
                print(f"  CYCLE {cycle} / {cycles}")
                print(f"{'='*60}")
            bot = TradingBot("BACKTEST")
            bot.run_backtest(days)
        return

    mode = choose_mode(arg_mode)

    if mode == "AUTO" and not config.BINANCE_TESTNET:
        from rich.console import Console
        from rich.prompt import Confirm
        Console().print(
            "\n[bold red]WARNING: AUTO mode with LIVE Binance Futures![/bold red]\n"
            "Real money will be at risk. Ensure you have tested in simulation first.\n"
        )
        if not Confirm.ask("Are you sure you want to continue?"):
            sys.exit(0)

    bot = TradingBot(mode)

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(sig, frame):
        bot._running = False

    _signal.signal(_signal.SIGINT,  _shutdown)
    _signal.signal(_signal.SIGTERM, _shutdown)

    bot.run()


if __name__ == "__main__":
    main()
