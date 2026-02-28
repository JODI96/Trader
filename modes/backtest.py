"""
modes/backtest.py — High-speed historical replay for NN training.

Downloads N days of 1m Binance Futures klines from the public production API
(no auth needed), then replays them through all indicators, the LSTM NN agent,
and the paper-trading simulation engine.

Position exits are evaluated against each candle's full high/low range via
SimulationMode.update_candle(), giving realistic results without tick-by-tick
data.

Usage:
    python main.py --mode backtest --days 30
    python main.py --mode backtest --days 7 --symbol ETHUSDT
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List

import requests

import config
from data.market_data import Candle

if TYPE_CHECKING:
    from main import TradingBot

logger = logging.getLogger(__name__)

# Binance Futures public REST — production data, no auth required
_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_KLINE_LIMIT = 1500   # Binance max per request


class BacktestRunner:
    """
    Replays historical 1m klines through the full bot pipeline:
      indicators → NN decision → signal → simulation → NN training.

    Key design decisions:
    - Position exit is checked BEFORE the new candle's NN decision, so a
      trade opened at candle N close is checked from candle N+1 onwards.
    - 15m HTF candles are delivered at the correct time (when their close_time
      precedes the current 1m candle's open_time).
    - Stdout logging is silenced to WARNING during replay to avoid 40k+ lines;
      everything still goes to trader.log.
    """

    def __init__(self, bot: "TradingBot") -> None:
        self._bot = bot
        self._checkpoints: List[dict] = []   # sampled every 1000 candles

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self, days: int) -> None:
        self._quiet_stdout(logging.ERROR)   # suppress LOSS/WIN warnings on stdout; file log keeps all
        # Use lower leverage so the paper account lasts longer per cycle.
        # NN reward is in R-multiples so learning is unaffected.
        orig_leverage = config.LEVERAGE
        config.LEVERAGE = config.BACKTEST_LEVERAGE
        try:
            self._run(days)
        finally:
            config.LEVERAGE = orig_leverage
            self._quiet_stdout(logging.INFO)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self, days: int) -> None:
        bot = self._bot
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 3600 * 1000

        print(f"\n{'='*60}")
        print(f"  BACKTEST  {config.SYMBOL}  {days} days")
        print(f"{'='*60}")

        print("  Fetching 1m klines from Binance…", end=" ", flush=True)
        klines_1m = self._fetch_klines("1m", start_ms, end_ms)
        print(f"{len(klines_1m):,} candles")

        print("  Fetching 15m klines from Binance…", end=" ", flush=True)
        klines_15m = self._fetch_klines("15m", start_ms, end_ms)
        print(f"{len(klines_15m):,} candles")

        # Print NN config so you can verify what's running
        nn = bot._nn_agent
        print(f"\n  NN CONFIG:")
        print(f"    State dim  : {nn._state_dim}")
        print(f"    LR         : {nn.LR}")
        print(f"    Buffer     : {nn.REPLAY_MAXLEN:,}  (currently {len(nn._buffer):,} filled)")
        print(f"    Batch      : {nn.BATCH_SIZE}")
        print(f"    Epsilon    : {nn.epsilon:.3f}  (floor={nn._eps_floor}  decay={nn.EPSILON_DECAY})")
        print(f"    Seq len    : {int(round(config.NN_SEQ_LEN))}  (max=32)")

        WARMUP = 200
        if len(klines_1m) < WARMUP + 1:
            print(f"  [ERROR] Not enough data ({len(klines_1m)} candles). "
                  f"Need at least {WARMUP + 1}. Aborting.")
            return

        # ── Warmup ───────────────────────────────────────────────────────────
        print(f"\n  Warming up {WARMUP} scalp + {min(50, len(klines_15m))} HTF candles…")
        for k in klines_1m[:WARMUP]:
            c = self._to_candle(k)
            bot.mds.scalp_candles.append(c)
            bot._warmup_indicators_scalp(c)

        htf_index = min(50, len(klines_15m))
        for k in klines_15m[:htf_index]:
            c = self._to_candle(k)
            bot.mds.htf_candles.append(c)
            bot._update_indicators_htf(c)

        # ── Replay ───────────────────────────────────────────────────────────
        replay = klines_1m[WARMUP:]
        total  = len(replay)
        print(f"  Replaying {total:,} candles…\n")

        t_start = time.time()
        balance_resets = 0

        for idx, k in enumerate(replay):
            candle = self._to_candle(k)

            # Auto-reset balance when depleted so the NN can keep learning.
            # Paper money is irrelevant during training — only the NN weights matter.
            if bot.risk_mgr.balance < bot.risk_mgr._initial_balance * 0.05:
                bot.risk_mgr.balance      = bot.risk_mgr._initial_balance
                bot.sim_mode.peak_balance = bot.risk_mgr._initial_balance
                balance_resets += 1
                print(
                    f"  [Balance reset #{balance_resets} — "
                    f"NN trades: {bot._nn_agent.stats()['trade_count']}  "
                    f"e={bot._nn_agent.epsilon:.3f}]",
                    flush=True,
                )

            # Deliver 15m candles whose close_time has passed
            while htf_index < len(klines_15m):
                htf_close_ms = int(klines_15m[htf_index][6])
                if htf_close_ms < candle.ts:
                    htf_c = self._to_candle(klines_15m[htf_index])
                    bot.mds.htf_candles.append(htf_c)
                    bot._update_indicators_htf(htf_c)
                    htf_index += 1
                else:
                    break

            # Check open position exit BEFORE processing new candle
            if bot.sim_mode.positions:
                closed = bot.sim_mode.update_candle(candle)
                if closed:
                    bot.dashboard.open_position_str = "No open position"
                    bot._record_nn_trade(closed.status, closed.close_price, candle.ts)

            # Feed candle through indicators and NN
            bot.mds.last_price = candle.close
            bot.mds.scalp_candles.append(candle)
            bot._on_scalp_candle(candle)

            # Immediately process any new signals
            bot._process_signals()

            # Progress every 1000 candles
            if (idx + 1) % 1000 == 0:
                self._print_progress(idx + 1, total, WARMUP, len(klines_1m), t_start, bot)

        # Close any still-open position at the last candle's close price
        if bot.sim_mode.positions:
            last_price = bot.mds.last_price
            bot.sim_mode.positions[0].status = "BE"
            bot.sim_mode._close_position(bot.sim_mode.positions[0], last_price, "BE")

        # ── Save and summarise ────────────────────────────────────────────────
        bot._nn_agent.save()
        bot._param_tuner.save()

        self._generate_plots(bot, days)
        self._print_summary(days, total, balance_resets, bot)

    def _print_progress(self, done: int, total: int, warmup: int,
                        n_klines: int, t_start: float, bot: "TradingBot") -> None:
        elapsed = time.time() - t_start
        rate    = done / elapsed if elapsed > 0 else 0
        eta_s   = (total - done) / rate if rate > 0 else 0
        pct     = done / total * 100

        stats = bot.sim_mode.stats()
        nn_st = bot._nn_agent.stats()
        tot   = stats['total_trades']
        wr    = stats['wins'] / tot * 100 if tot > 0 else 0.0
        loss_str = f"{nn_st['last_loss']:.4f}" if nn_st['last_loss'] is not None else "n/a"

        # Visual epsilon bar: shows how far epsilon has decayed toward the floor
        # Full bar = at floor (done exploring), empty = just started
        BAR = 20
        filled   = int(nn_st['eps_pct'] / 100 * BAR)
        eps_bar  = '#' * filled + '.' * (BAR - filled)
        eps_str  = f"e:{nn_st['epsilon']:.3f} [{eps_bar}] {nn_st['eps_pct']:.0f}%"

        print(
            f"  {pct:5.1f}%  [{done + warmup:,}/{n_klines:,}]  "
            f"trades:{tot}  W/L:{stats['wins']}/{stats['losses']}({wr:.0f}%)  "
            f"bal:${stats['balance']:,.0f}  PnL:{stats['session_pnl']:+.2f}"
            f"         {eps_str}  loss:{loss_str}  "
            f"buf:{nn_st['buf_size']:,}/{nn_st['buf_capacity']:,}({nn_st['buf_pct']:.0f}%)  "
            f"ETA:{eta_s/60:.1f}m",
            flush=True,
        )
        # Store snapshot for post-run plots
        self._checkpoints.append({
            'pct':          pct,
            'trade_count':  nn_st['trade_count'],
            'epsilon':      nn_st['epsilon'],
            'loss':         nn_st['last_loss'],
            'win_rate':     wr / 100.0,
            'buf_pct':      nn_st['buf_pct'],
            'total_reward': nn_st['total_reward'],
        })

    def _print_summary(self, days: int, total: int, balance_resets: int, bot: "TradingBot") -> None:
        stats = bot.sim_mode.stats()
        nn_st = bot._nn_agent.stats()
        tuned = bot._param_tuner.current_values()

        print(f"\n{'='*60}")
        print(f"  BACKTEST COMPLETE — {days} days of {config.SYMBOL}")
        print(f"{'='*60}")
        print(f"  Candles replayed : {total:,}")
        if balance_resets:
            print(f"  Balance resets   : {balance_resets}  (training artifact — expected)")
        print(f"  Total trades     : {stats['total_trades']}")
        print(f"  W / L / BE       : {stats['wins']} / {stats['losses']} / {stats['be']}")
        print(f"  Win rate         : {stats['win_rate']:.1%}")
        print(f"  Net PnL          : {stats['session_pnl']:+.2f} USDT")
        print(f"  Fees paid        : -{stats['fees_paid']:.4f} USDT")
        print(f"  Max drawdown     : {stats['max_drawdown']:.1%}")
        print(f"  Final balance    : ${stats['balance']:,.2f} USDT")
        print(f"  NN trades        : {nn_st['trade_count']}")
        print(f"  NN epsilon       : {nn_st['epsilon']:.3f}")
        print(f"  NN total reward  : {nn_st['total_reward']:+.2f}")
        loss_str = f"{nn_st['last_loss']:.4f}" if nn_st["last_loss"] is not None else "n/a"
        print(f"  NN last loss     : {loss_str}")
        print(f"  Seq length       : {nn_st['seq_len']}")
        print(f"  Tuned params     :")
        for name, val in tuned.items():
            print(f"    {name:<22} {val:.4f}")
        print(f"{'='*60}\n")
        print("  NN model saved to ml/model/agent.pt")
        print("  Run 'python main.py --mode sim' to trade with the trained model.\n")

    # ── Analysis plots ───────────────────────────────────────────────────────

    def _generate_plots(self, bot: "TradingBot", days: int) -> None:
        """Generate a multi-panel PNG analysis chart after the backtest."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("  [charts] matplotlib not installed — run: pip install matplotlib")
            return

        closed = bot.sim_mode.closed
        if len(closed) < 5:
            print("  [charts] Too few trades to plot — skipping")
            return

        out_dir = Path(__file__).parent.parent / "ml" / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"backtest_{days}d_{ts_str}.png"

        # ── Derive per-trade metadata ─────────────────────────────────────────
        def _session(hour_frac: float) -> str:
            if   13.5 <= hour_frac < 17.0: return "LN/NY"
            elif  8.0 <= hour_frac < 13.5: return "London"
            elif 17.0 <= hour_frac < 22.0: return "NY"
            else:                           return "Asia"

        SESSIONS  = ["Asia", "London", "LN/NY", "NY"]
        WEEKDAYS  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        sess_wins  = {s: 0 for s in SESSIONS}
        sess_total = {s: 0 for s in SESSIONS}
        day_wins   = {d: 0 for d in WEEKDAYS}
        day_total  = {d: 0 for d in WEEKDAYS}
        hour_wins  = {h: 0 for h in range(24)}
        hour_total = {h: 0 for h in range(24)}
        rewards, directions = [], []

        for pos in closed:
            dt  = datetime.fromtimestamp(pos.open_time, tz=timezone.utc)
            hf  = dt.hour + dt.minute / 60.0
            ses = _session(hf)
            wd  = WEEKDAYS[dt.weekday()]
            win = pos.status == "WIN"

            sess_total[ses] += 1
            day_total[wd]   += 1
            hour_total[dt.hour] += 1
            if win:
                sess_wins[ses] += 1
                day_wins[wd]   += 1
                hour_wins[dt.hour] += 1

            rewards.append(pos.pnl_usdt)
            directions.append(pos.direction)

        # ── Checkpoint time-series ────────────────────────────────────────────
        cp = self._checkpoints
        cp_trades   = [c["trade_count"]  for c in cp]
        cp_eps      = [c["epsilon"]      for c in cp]
        cp_loss     = [c["loss"] if c["loss"] is not None else float("nan") for c in cp]
        cp_wr       = [c["win_rate"]     for c in cp]
        cp_reward   = [c["total_reward"] for c in cp]

        # ── Layout ───────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 14))
        fig.patch.set_facecolor("#1a1a2e")
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        TITLE_CLR  = "#e0e0e0"
        AXIS_CLR   = "#888888"
        GRID_CLR   = "#333355"
        WIN_CLR    = "#2ecc71"
        LOSS_CLR   = "#e74c3c"
        EPS_CLR    = "#f39c12"
        LOSS_LINE  = "#3498db"
        REWARD_CLR = "#9b59b6"

        def _style(ax, title):
            ax.set_facecolor("#0f0f1e")
            ax.set_title(title, color=TITLE_CLR, fontsize=10, pad=6)
            ax.tick_params(colors=AXIS_CLR, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)
            ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--")

        fig.suptitle(
            f"Backtest ML Analysis  —  {days} days  |  {len(closed)} trades  |  "
            f"LR={bot._nn_agent.LR}  dim={bot._nn_agent._state_dim}",
            color=TITLE_CLR, fontsize=13, y=0.98,
        )

        # 1. Epsilon + Loss over time
        ax1 = fig.add_subplot(gs[0, :2])
        _style(ax1, "Epsilon Decay  &  Training Loss")
        if cp_trades:
            ax1.plot(cp_trades, cp_eps, color=EPS_CLR, linewidth=1.5, label="Epsilon")
            ax1.set_ylabel("Epsilon", color=EPS_CLR, fontsize=8)
            ax1.tick_params(axis="y", labelcolor=EPS_CLR)
            ax1.axhline(bot._nn_agent._eps_floor, color=EPS_CLR,
                        linestyle=":", linewidth=1, alpha=0.5, label="ε floor")
            ax2 = ax1.twinx()
            ax2.set_facecolor("#0f0f1e")
            ax2.plot(cp_trades, cp_loss, color=LOSS_LINE, linewidth=1.2,
                     alpha=0.85, label="Loss")
            ax2.set_ylabel("Loss", color=LOSS_LINE, fontsize=8)
            ax2.tick_params(axis="y", labelcolor=LOSS_LINE, labelsize=8)
            lines1, lbl1 = ax1.get_legend_handles_labels()
            lines2, lbl2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, lbl1 + lbl2,
                       facecolor="#1a1a2e", edgecolor=GRID_CLR,
                       labelcolor=TITLE_CLR, fontsize=7, loc="upper right")
        ax1.set_xlabel("Trades", color=AXIS_CLR, fontsize=8)

        # 2. Rolling win rate over time
        ax3 = fig.add_subplot(gs[0, 2])
        _style(ax3, "Win Rate Over Time")
        if cp_trades:
            ax3.plot(cp_trades, [w * 100 for w in cp_wr],
                     color=WIN_CLR, linewidth=1.5)
            ax3.axhline(25, color="#888888", linestyle="--",
                        linewidth=1, label="25% breakeven")
            ax3.axhline(30, color=WIN_CLR, linestyle=":",
                        linewidth=1, alpha=0.6, label="30% target")
            ax3.set_ylabel("Win Rate %", color=AXIS_CLR, fontsize=8)
            ax3.set_xlabel("Trades", color=AXIS_CLR, fontsize=8)
            ax3.set_ylim(0, 60)
            ax3.legend(facecolor="#1a1a2e", edgecolor=GRID_CLR,
                       labelcolor=TITLE_CLR, fontsize=7)

        # 3. Win rate by session
        ax4 = fig.add_subplot(gs[1, 0])
        _style(ax4, "Win Rate by Session")
        sess_wr  = [sess_wins[s] / sess_total[s] * 100 if sess_total[s] > 0 else 0
                    for s in SESSIONS]
        sess_cnt = [sess_total[s] for s in SESSIONS]
        bars = ax4.bar(SESSIONS, sess_wr,
                       color=[WIN_CLR if w >= 25 else LOSS_CLR for w in sess_wr],
                       alpha=0.8)
        ax4.axhline(25, color="#888888", linestyle="--", linewidth=1)
        ax4.set_ylabel("Win Rate %", color=AXIS_CLR, fontsize=8)
        ax4.set_ylim(0, 60)
        for bar, cnt in zip(bars, sess_cnt):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"n={cnt}", ha="center", va="bottom",
                     color=AXIS_CLR, fontsize=7)

        # 4. Win rate by weekday
        ax5 = fig.add_subplot(gs[1, 1])
        _style(ax5, "Win Rate by Weekday")
        day_wr  = [day_wins[d] / day_total[d] * 100 if day_total[d] > 0 else 0
                   for d in WEEKDAYS]
        day_cnt = [day_total[d] for d in WEEKDAYS]
        bars = ax5.bar(WEEKDAYS, day_wr,
                       color=[WIN_CLR if w >= 25 else LOSS_CLR for w in day_wr],
                       alpha=0.8)
        ax5.axhline(25, color="#888888", linestyle="--", linewidth=1)
        ax5.set_ylabel("Win Rate %", color=AXIS_CLR, fontsize=8)
        ax5.set_ylim(0, 60)
        for bar, cnt in zip(bars, day_cnt):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"n={cnt}", ha="center", va="bottom",
                     color=AXIS_CLR, fontsize=7)

        # 5. Win rate by hour (UTC)
        ax6 = fig.add_subplot(gs[1, 2])
        _style(ax6, "Win Rate by Hour (UTC)")
        hours   = list(range(24))
        hour_wr = [hour_wins[h] / hour_total[h] * 100 if hour_total[h] > 0 else 0
                   for h in hours]
        ax6.bar(hours, hour_wr,
                color=[WIN_CLR if w >= 25 else LOSS_CLR for w in hour_wr],
                alpha=0.8)
        ax6.axhline(25, color="#888888", linestyle="--", linewidth=1)
        ax6.set_xlabel("Hour UTC", color=AXIS_CLR, fontsize=8)
        ax6.set_ylabel("Win Rate %", color=AXIS_CLR, fontsize=8)
        ax6.set_xticks(range(0, 24, 3))
        ax6.set_ylim(0, 70)
        # Session shading
        for start, end, lbl in [(0, 8, "Asia"), (8, 13.5, "LN"), (13.5, 17, "OV"), (17, 22, "NY")]:
            ax6.axvspan(start, end, alpha=0.07, color=WIN_CLR)

        # 6. Reward distribution
        ax7 = fig.add_subplot(gs[2, 0])
        _style(ax7, "PnL Distribution per Trade")
        wins_r  = [r for r, pos in zip(rewards, closed) if pos.status == "WIN"]
        loss_r  = [r for r, pos in zip(rewards, closed) if pos.status == "LOSS"]
        if wins_r:
            ax7.hist(wins_r,  bins=30, color=WIN_CLR,  alpha=0.7, label="WIN")
        if loss_r:
            ax7.hist(loss_r,  bins=30, color=LOSS_CLR, alpha=0.7, label="LOSS")
        ax7.axvline(0, color="#888888", linewidth=1)
        ax7.set_xlabel("PnL (USDT)", color=AXIS_CLR, fontsize=8)
        ax7.set_ylabel("Count", color=AXIS_CLR, fontsize=8)
        ax7.legend(facecolor="#1a1a2e", edgecolor=GRID_CLR,
                   labelcolor=TITLE_CLR, fontsize=7)

        # 7. Cumulative reward
        ax8 = fig.add_subplot(gs[2, 1])
        _style(ax8, "Cumulative NN Reward")
        if cp_trades:
            ax8.plot(cp_trades, cp_reward, color=REWARD_CLR, linewidth=1.5)
            ax8.axhline(0, color="#888888", linewidth=1, linestyle="--")
            ax8.fill_between(cp_trades, cp_reward, 0,
                             where=[r >= 0 for r in cp_reward],
                             color=WIN_CLR, alpha=0.15)
            ax8.fill_between(cp_trades, cp_reward, 0,
                             where=[r < 0 for r in cp_reward],
                             color=LOSS_CLR, alpha=0.15)
            ax8.set_xlabel("Trades", color=AXIS_CLR, fontsize=8)
            ax8.set_ylabel("Total Reward (R)", color=AXIS_CLR, fontsize=8)

        # 8. LONG vs SHORT win rate
        ax9 = fig.add_subplot(gs[2, 2])
        _style(ax9, "LONG vs SHORT Performance")
        long_w  = sum(1 for p in closed if p.direction == "LONG"  and p.status == "WIN")
        long_t  = sum(1 for p in closed if p.direction == "LONG")
        short_w = sum(1 for p in closed if p.direction == "SHORT" and p.status == "WIN")
        short_t = sum(1 for p in closed if p.direction == "SHORT")
        lbls  = ["LONG", "SHORT"]
        wrs   = [long_w / long_t * 100 if long_t > 0 else 0,
                 short_w / short_t * 100 if short_t > 0 else 0]
        cnts  = [long_t, short_t]
        bars  = ax9.bar(lbls, wrs,
                        color=[WIN_CLR if w >= 25 else LOSS_CLR for w in wrs],
                        alpha=0.8, width=0.4)
        ax9.axhline(25, color="#888888", linestyle="--", linewidth=1)
        ax9.set_ylabel("Win Rate %", color=AXIS_CLR, fontsize=8)
        ax9.set_ylim(0, 60)
        for bar, cnt in zip(bars, cnts):
            ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"n={cnt}", ha="center", va="bottom",
                     color=AXIS_CLR, fontsize=8)

        plt.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  [CHART] Analysis saved -> {out_path}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _fetch_klines(self, interval: str, start_ms: int, end_ms: int) -> list:
        """
        Paginated fetch from Binance Futures public API.
        Always uses the production endpoint regardless of BINANCE_TESTNET,
        because the testnet has very limited historical data.
        """
        all_klines: list = []
        current_start = start_ms

        while current_start < end_ms:
            params = {
                "symbol":    config.SYMBOL,
                "interval":  interval,
                "limit":     _KLINE_LIMIT,
                "startTime": current_start,
                "endTime":   end_ms,
            }
            for attempt in range(3):
                try:
                    resp = requests.get(_KLINES_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    batch = resp.json()
                    break
                except Exception as e:
                    logger.warning(f"klines fetch error (attempt {attempt+1}/3): {e}")
                    time.sleep(5)
            else:
                logger.error("Failed to fetch klines after 3 attempts — aborting pagination")
                break

            if not batch:
                break

            all_klines.extend(batch)
            last_close_ms = int(batch[-1][6])
            current_start = last_close_ms + 1

            if len(batch) < _KLINE_LIMIT:
                break   # last page

            time.sleep(0.1)   # rate-limit courtesy

        return all_klines

    @staticmethod
    def _to_candle(k) -> Candle:
        """Convert a Binance kline list to a Candle dataclass."""
        vol      = float(k[5])
        buy_vol  = float(k[9])   # taker buy base asset volume (real delta)
        sell_vol = vol - buy_vol
        return Candle(
            ts         = int(k[0]),
            open       = float(k[1]),
            high       = float(k[2]),
            low        = float(k[3]),
            close      = float(k[4]),
            volume     = vol,
            buy_vol    = buy_vol,
            sell_vol   = sell_vol,
            delta      = buy_vol - sell_vol,
            num_trades = int(k[8]),
        )

    @staticmethod
    def _quiet_stdout(level: int) -> None:
        """Adjust the stdout log handler level (suppresses INFO during replay)."""
        for h in logging.getLogger().handlers:
            if (isinstance(h, logging.StreamHandler)
                    and getattr(h, "stream", None) is sys.stdout):
                h.setLevel(level)
