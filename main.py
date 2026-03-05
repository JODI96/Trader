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
import math
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
    con.print("  [red]3[/red]  AUTO        — live Binance Futures trading")
    con.print("  [magenta]4[/magenta]  FETCH       — download historical data for offline backtest\n")

    choice = Prompt.ask("Select mode", choices=["1", "2", "3", "4"], default="1")
    return {"1": "SIMULATION", "2": "ALERT", "3": "AUTO", "4": "FETCH"}[choice]


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
        self.filters  = TradeFilters(self.atr, self.dv, self.calendar,
                                     backtest_mode=(mode == "BACKTEST"))

        # ── Modes ───────────────────────────────────────────────────────────
        self.sim_mode   = SimulationMode(self.risk_mgr)
        self.alert_mode = AlertMode()

        self.auto_mode: Optional[AutoTradeMode] = None
        if mode == "AUTO":
            self.auto_mode = AutoTradeMode(self.client, self.risk_mgr)

        # ── Neural Network Agent ─────────────────────────────────────────────
        self._nn_agent = DQNAgent(state_dim=STATE_DIM, auto_mode=(mode == "AUTO"))
        # Inference-only in live modes — model weights are never updated
        if mode in ("SIMULATION", "AUTO", "ALERT"):
            self._nn_agent.set_training(False)
        # Pending NN experience for the currently open position
        self._pending_nn_state:  Optional[list]  = None
        self._pending_nn_action: Optional[int]   = None
        self._pending_nn_rr:     float           = 3.0   # default R:R for reward calc
        # Previous candle state — needed to record hold experiences
        self._prev_state:        Optional[list]  = None
        # Cooldown: candles remaining before next trade is allowed
        self._nn_cooldown:       int             = 0
        # Intra-trade reward shaping accumulator (reset on each trade open/close)
        self._nn_shaping_acc:    float           = 0.0
        self._nn_t_in_trade:     int             = 0
        self._nn_entry_price:    float           = 0.0
        self._nn_prev_close:     float           = 0.0
        self._nn_prev_vol_ratio: float           = 0.0
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

        # ── Load persisted simulation state (if any) ────────────────────────
        if mode == "SIMULATION":
            saved_log = self.sim_mode.load_state()
            for entry in saved_log:
                self.dashboard._signal_log.append(entry)

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
            self.mds.seed_from_klines(scalp_klines, "1m")   # 1m MTF = same data

            htf_klines = self.client.get_klines(
                config.SYMBOL, "15m", limit=100
            )
            self.mds.seed_from_klines(htf_klines, "htf")

            mtf_5m_klines = self.client.get_klines(
                config.SYMBOL, "5m", limit=config.MTF_5M_LOOKBACK
            )
            self.mds.seed_from_klines(mtf_5m_klines, "5m")

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
        self.client.start_obi_polling(config.SYMBOL, self.mds)
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
        if self.mode == "SIMULATION":
            self.sim_mode.save_state(list(self.dashboard._signal_log))
        logger.info("Bot stopped.")

    def run_backtest(self, days: int, data_key: Optional[str] = None) -> None:
        """
        High-speed historical replay for NN training.
        When data_key is set, loads from local data/historical/ instead of the API.
        Otherwise downloads `days` days of klines from Binance Futures.
        The WebSocket is never started; no Rich dashboard is shown.
        """
        from modes.backtest import BacktestRunner
        if data_key:
            logger.info(f"Starting backtest: {config.SYMBOL} from local dataset '{data_key}'")
        else:
            logger.info(f"Starting backtest: {days} days of {config.SYMBOL}")
        BacktestRunner(self).run(days, data_key)
        logger.info("Backtest complete.")

    # ── Candle handlers ──────────────────────────────────────────────────────

    def _on_scalp_candle(self, candle) -> None:
        self._update_indicators_scalp(candle)

        # Accumulate intra-trade shaping for any already-open position.
        # Must run after indicators (needs fresh dv.last_volume / avg_volume)
        # but before the NN decision so the trade-open flag isn't set yet.
        if self._pending_nn_state is not None and self._pending_nn_action is not None:
            self._accumulate_intra_shaping(candle)

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
                # HOLD reward is shaped by confluence so the model can't exploit
                # by doing nothing:
                #   low conf  (<0.35) → small reward for correctly sitting out
                #   mid conf  (0.35–0.60) → neutral
                #   high conf (>0.60) → penalty for missing a good setup
                long_conf  = compute_confluence(current_state, "LONG")
                short_conf = compute_confluence(current_state, "SHORT")
                best_conf  = max(long_conf, short_conf)
                if best_conf < 0.35:
                    hold_reward =  0.05   # good: no edge, stay flat
                elif best_conf > 0.60:
                    hold_reward = -0.20   # bad: clear setup, should have traded
                else:
                    hold_reward =  0.0
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
        # Require minimum confluence (35%) so the model only trades in states where classical
        # indicators partially agree — filters out the noisiest signals.
        _entry_dir  = "LONG" if nn_action == LONG else "SHORT"
        _entry_conf = compute_confluence(current_state, _entry_dir) if nn_action != HOLD else 0.0
        if ok and nn_action != HOLD and self._pending_nn_state is None and self._nn_cooldown == 0 and _entry_conf >= 0.35:
            sig = self._build_nn_signal(nn_action, candle)
            if sig:
                self._nn_agent.snapshot_open_seq()
                self._pending_nn_state  = current_state
                self._pending_nn_action = nn_action
                self._pending_nn_rr     = sig.rr_ratio
                self._prev_state        = None   # consumed by this trade
                # --- intra-trade shaping init ---
                self._nn_entry_price    = candle.close
                self._nn_prev_close     = candle.close
                self._nn_t_in_trade     = 0
                self._nn_shaping_acc    = 0.0
                avg_vol = self.dv.avg_volume
                self._nn_prev_vol_ratio = (
                    self.dv.last_volume / avg_vol if avg_vol > 0 else 0.0
                )
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

    def _accumulate_intra_shaping(self, candle) -> None:
        """
        Called once per candle while a trade is open (before the NN decision).
        Computes per-candle directional bonuses/penalties and accumulates them
        into self._nn_shaping_acc, which is added to the outcome reward at close.

        Tuning constants
        ----------------
        K        : exp-decay rate for speed/vol bonuses (larger = bonuses fade faster)
        V_MIN    : minimum directional price move this candle to qualify for speed bonus
        A_MIN    : minimum vol_accel (vol_ratio change vs prev candle) for vol signals
        VOL_PEN  : penalty magnitude when vol surges in the wrong direction
        CONSOL   : per-candle consolidation penalty (suppressed on strong moves)
        CAP      : hard cap on accumulator magnitude to prevent swamping base reward
        """
        K        = 0.3   # decay: ~74% at t=1, ~22% at t=5, negligible by t=10
        V_MIN    = 0.0   # any positive candle in the right direction qualifies
        A_MIN    = 0.3   # vol_ratio must jump by 0.3 (= 30% of 3× avg) to count
        VOL_PEN  = 3.0   # wrong-way volume penalty
        CONSOL   = 0.2   # consolidation penalty per candle
        CAP      = 15.0  # accumulator hard cap (±)

        dir_sign = 1.0 if self._pending_nn_action == LONG else -1.0

        price_progress = dir_sign * (candle.close - self._nn_entry_price)
        price_velocity = dir_sign * (candle.close - self._nn_prev_close)

        avg_vol = self.dv.avg_volume
        cur_vol_ratio = self.dv.last_volume / avg_vol if avg_vol > 0 else 0.0
        vol_accel = cur_vol_ratio - self._nn_prev_vol_ratio

        t     = self._nn_t_in_trade
        decay = math.exp(-K * t)

        # SpeedBonus: price moved in right direction this candle AND net progress positive
        speed_bonus = 3.0 * decay if (price_progress > 0 and price_velocity > V_MIN) else 0.0

        # VolBonus: volume accelerating AND price is net-positive for our direction
        vol_bonus = 3.0 * decay if (vol_accel > A_MIN and price_progress > 0) else 0.0

        # VolWrongPen: volume surging while price moves against us
        vol_wrong_pen = -VOL_PEN if (vol_accel > A_MIN and price_progress < 0) else 0.0

        # ConsolPen: price not making meaningful net progress (suppressed on strong moves)
        atr_threshold = 0.5 * self.atr.atr if self.atr.is_ready else 0.0
        consol_pen = 0.0 if abs(price_progress) > atr_threshold else -CONSOL

        step = speed_bonus + vol_bonus + vol_wrong_pen + consol_pen
        self._nn_shaping_acc = max(-CAP, min(CAP, self._nn_shaping_acc + step))

        # Advance tracking state
        self._nn_prev_close     = candle.close
        self._nn_prev_vol_ratio = cur_vol_ratio
        self._nn_t_in_trade    += 1

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

        # Base reward.
        #   WIN  (+10): TP hit — full reward.
        #   BE   (+2):  Break-even stop triggered — direction was right, price moved
        #               1R in favour before reversing.  Positive reward incentivises
        #               the model to find trades that at least reach the 1:1 level.
        #   LOSS (-4):  SL hit before 1R was ever reached — pure bad entry.
        # Break-even win rate (WIN vs LOSS only) = 4/(10+4) ≈ 28.6%.
        if outcome == "WIN":
            base_reward = 10.0
        elif outcome == "BE":
            base_reward = 2.0
        else:   # LOSS — price never reached 1R in our direction
            base_reward = -4.0

        direction  = "LONG" if self._pending_nn_action == LONG else "SHORT"
        confluence = compute_confluence(self._pending_nn_state, direction)

        if self.mode == "BACKTEST":
            # No confluence shaping — 1m candle confluence is unreliable.
            # Intra-trade shaping carries all quality signal.
            shaped_reward = base_reward
        else:
            # Confluence-shaped reward for live modes (15s candles).
            if base_reward > 0:
                shaped_reward = base_reward * (0.7 + 0.6 * confluence)
            else:
                shaped_reward = base_reward * (1.3 - 0.6 * confluence)

        # Add accumulated intra-trade shaping — clamped by outcome:
        #   WIN/BE: only positive shaping (reward good in-trade movement)
        #   LOSS:   only negative shaping (punish bad entries harder)
        if outcome in ("WIN", "BE"):
            shaped_reward += max(0.0, self._nn_shaping_acc)
        else:
            shaped_reward += min(0.0, self._nn_shaping_acc)

        ts_ms = close_ts_ms if close_ts_ms > 0 else int(self.mds.last_trade_ts * 1000)
        close_state = build_state(self.vwap, self.cvd, self.dv,
                                  self.atr, self.vp, self.mds, self.sr, ts_ms)

        self._nn_agent.record_trade(
            self._pending_nn_state,
            self._pending_nn_action,
            shaped_reward,
            close_state,
        )


        st = self._nn_agent.stats()
        loss_str = f"{st['last_loss']:.4f}" if st["last_loss"] is not None else "n/a"
        shaping_applied = max(0.0, self._nn_shaping_acc) if outcome == "WIN" else min(0.0, self._nn_shaping_acc)
        logger.info(
            f"NN trained: {outcome} reward={shaped_reward:+.2f} "
            f"(base={base_reward:+.1f} shaping={shaping_applied:+.2f}/{self._nn_shaping_acc:+.2f} conf={confluence:.2f})  "
            f"loss={loss_str}  e={st['epsilon']:.3f}  trades={st['trade_count']}"
        )

        # Reset pending experience and start cooldown
        self._pending_nn_state  = None
        self._pending_nn_action = None
        self._nn_cooldown       = config.NN_TRADE_COOLDOWN
        self._nn_shaping_acc    = 0.0
        self._nn_t_in_trade     = 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orderflow Scalping Bot")
    p.add_argument("--mode", choices=["sim", "simulation", "alert", "auto", "backtest", "validate", "fetch"],
                   help="Trading mode (default: interactive prompt)")
    p.add_argument("--symbol", default=None,
                   help="Override symbol (e.g. ETHUSDT)")
    p.add_argument("--days", type=int, default=None,
                   help="Days of history for backtest/fetch mode (default: BACKTEST_DAYS in config)")
    p.add_argument("--cycles", type=int, default=1,
                   help="How many times to repeat the backtest (default: 1)")
    p.add_argument("--year", type=int, default=None,
                   help="Year for fetch or backtest (e.g. 2025)")
    p.add_argument("--month", type=int, default=None,
                   help="Month for fetch or backtest (1–12). Use together with --year.")
    p.add_argument("--trim-buffer", type=int, default=None,
                   metavar="N",
                   help="Remove the oldest N experiences from the replay buffer and exit.")
    p.add_argument("--flush-bad", action="store_true", default=False,
                   help="Remove all experiences recorded when win rate was under 50%% and exit.")
    p.add_argument("--set-epsilon", type=float, default=None,
                   metavar="E",
                   help="Override saved epsilon (e.g. 0.8 to re-enable exploration) and exit.")
    p.add_argument("--data", default=None,
                   metavar="KEY",
                   help=(
                       "Use a local dataset for backtest. "
                       "KEY: 'year' (all years), 'YYYY-MM' (e.g. 2025-01), "
                       "or season name (spring / summer / autumn / winter). "
                       "Prefer --year / --month for explicit control."
                   ))
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
    elif args.mode == "validate":
        arg_mode = "VALIDATE"
    elif args.mode == "fetch":
        arg_mode = "FETCH"

    # Epsilon override — load model, set epsilon, save, exit
    if args.set_epsilon is not None:
        from ml.neural_agent import DQNAgent
        agent = DQNAgent(state_dim=STATE_DIM)
        old = agent.epsilon
        agent.epsilon = max(0.0, min(1.0, args.set_epsilon))
        agent.save()
        print(f"Epsilon updated: {old:.3f} -> {agent.epsilon:.3f}  (saved to ml/model/agent.pt)")
        return

    # Buffer operations — standalone, no bot needed
    if args.trim_buffer is not None or args.flush_bad:
        from ml.neural_agent import DQNAgent
        agent  = DQNAgent(state_dim=STATE_DIM)
        before = len(agent._buffer)
        if args.flush_bad:
            removed = agent.flush_bad()
            bad_pct = removed / before * 100 if before > 0 else 0
            print(f"Flushed bad experiences: {before:,} -> {before - removed:,} ({removed:,} removed, {bad_pct:.1f}% were bad)")
        if args.trim_buffer is not None:
            removed = agent.trim_buffer(args.trim_buffer)
            after   = len(agent._buffer)
            print(f"Buffer trimmed: {before:,} -> {after:,} ({removed:,} removed)")
        return

    # Fetch mode: download and save historical data, then exit
    if arg_mode == "FETCH":
        from modes.fetch_data import DataFetcher
        DataFetcher(symbol=args.symbol).run(
            days=args.days,
            year=args.year,
            month=args.month,
        )
        return

    # Backtest / Validate — both replay historical data, validate skips learning
    if arg_mode in ("BACKTEST", "VALIDATE"):
        days = args.days if args.days is not None else config.BACKTEST_DAYS

        # Build data_key: --year/--month take priority over --data
        if args.year and args.month:
            data_key = f"{args.year}-{args.month:02d}"
        elif args.year:
            data_key = str(args.year)
        else:
            data_key = args.data if args.data else None

        cycles = max(1, args.cycles)
        for cycle in range(1, cycles + 1):
            if cycles > 1:
                print(f"\n{'='*60}")
                print(f"  CYCLE {cycle} / {cycles}")
                print(f"{'='*60}")
            bot = TradingBot("BACKTEST")
            if arg_mode == "VALIDATE":
                bot._nn_agent.set_training(False)
            bot.run_backtest(days, data_key)
        return

    mode = choose_mode(arg_mode)

    # Fetch mode can also be selected interactively
    if mode == "FETCH":
        from modes.fetch_data import DataFetcher
        DataFetcher(symbol=args.symbol).run(
            days=args.days,
            year=args.year,
            month=args.month,
        )
        return

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
