"""
ui/dashboard.py — Rich Live terminal dashboard.

Layout (updates every 500ms):
┌─ BTCUSDT FUTURES  15s  MODE: SIMULATION ────────────────────────────────────┐
│ PRICE           │ INDICATORS                  │ POSITION / LAST SIGNAL       │
│ $52,340.00      │ VWAP    $52,280  (+0.1σ)   │ No open position              │
│ Δ  +120.50      │ Upper1  $52,800             │ Last: LONG @52300             │
│ Vol  2.3× avg   │ Lower1  $51,760             │ SL: $52,200   TP: $52,600    │
│ BUBBLE [BUY]    │ ATR     $45.00  (0.09%)     │ Conf: ██░  Strategy: Absorb   │
│ CVD   +1,245    │ POC     $52,100             │                               │
│ Zone  NEUTRAL   │ VAH     $52,800             │                               │
│                 │ VAL     $51,600             │                               │
├─────────────────┴─────────────────────────────┴───────────────────────────────┤
│ SESSION: 2 trades │ 1W  1L │ PnL: +$23.40 │ Balance: $1,023  │ Losses: 2/3  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ FILTERS: Vol ✓ │ News ✓ │ Session ✓ │ Volatility ✓                          │
│ NEWS: ECB Speech  14:30 UTC  (in 2h 15m)  ⚠ HIGH                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [12:34:15] LONG @52300 — Absorption (delta 2.3×, body 0.05%)  Conf:3/3       │
│ [12:28:40] SHORT @52580 — Delta Divergence  (closed WIN +$18.20)              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  15s CHART                                                                     │
│  $84,200 |  ||  |  || ||  | |                                                  │
│          | |||  |  |||||| | |                                                  │
│          | |||  | ||||||| | |                                                  │
│  $84,000  ||||  | |||||||^| |                         (^ = LONG  v = SHORT)   │
│           ||||  |v||||||||| |                                                  │
│            |||  ||||||||||||                                                   │
│  $83,800    ||   |||||||||||                                                   │
│  HH:MM                                HH:MM UTC                               │
└─────────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Deque, List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config

if TYPE_CHECKING:
    from data.economic_calendar import EconomicCalendar, EconomicEvent
    from data.market_data import Signal
    from indicators.atr import ATRIndicator
    from indicators.cvd import CVDIndicator
    from indicators.delta_volume import DeltaVolumeIndicator
    from indicators.support_resistance import SupportResistance
    from indicators.volume_profile import VolumeProfile
    from indicators.vwap import VWAPIndicator
    from strategy.filters import TradeFilters
    from strategy.risk_manager import RiskManager
    from ml.neural_agent import DQNAgent
    from ml.param_tuner import ParameterTuner


_CONF_CHARS = {0: "---", 1: "#--", 2: "##-", 3: "###"}


# ── Candle chart ─────────────────────────────────────────────────────────────

@dataclass
class _ChartTrade:
    candle_ts: int          # candle bucket ts_ms when trade opened
    direction: str          # LONG | SHORT
    price:     float
    outcome:   Optional[str] = None   # WIN | LOSS | BE | None (open)


class CandleChart:
    """
    Rolling buffer of 15s candles + trade entry markers.
    Renders an ASCII candlestick chart sized to fit the available terminal width.
    """
    MAX_CANDLES  = 120
    CHART_HEIGHT = 12   # price rows
    MAX_TRADES   = 30

    def __init__(self) -> None:
        self._candles: Deque = deque(maxlen=self.MAX_CANDLES)
        self._trades:  List[_ChartTrade] = []

    def add_candle(self, candle) -> None:
        self._candles.append(candle)

    def open_trade(self, candle_ts: int, direction: str, price: float) -> None:
        if len(self._trades) >= self.MAX_TRADES:
            self._trades.pop(0)
        self._trades.append(_ChartTrade(candle_ts, direction, price))

    def close_trade(self, outcome: str) -> None:
        for t in reversed(self._trades):
            if t.outcome is None:
                t.outcome = outcome
                break

    def render(self, width: int = 100) -> Text:
        """
        Render the chart into a Rich Text object.
        `width` is the available content width (chars).
        """
        candles = list(self._candles)
        if len(candles) < 4:
            return Text("  Chart warming up...", style="dim")

        H = self.CHART_HEIGHT
        Y_LABEL_W = 10   # chars reserved for right-side price labels
        chart_cols = max(10, width - Y_LABEL_W - 2)  # subtract margin + Y labels
        # Show only as many candles as fit
        if len(candles) > chart_cols:
            candles = candles[-chart_cols:]
        W = len(candles)

        # Price range
        p_max = max(c.high for c in candles)
        p_min = min(c.low  for c in candles)
        p_rng = p_max - p_min
        if p_rng < 1e-8:
            p_rng = 1.0
        pad   = p_rng * 0.06
        p_max += pad
        p_min -= pad
        p_rng  = p_max - p_min

        def to_row(price: float) -> int:
            r = int((p_max - price) / p_rng * (H - 1))
            return max(0, min(H - 1, r))

        # Trade lookup by candle_ts
        trade_map: dict = {}
        for tr in self._trades:
            trade_map[tr.candle_ts] = tr

        # Build grid[row][col] = (char, style)
        grid = [[(' ', '') for _ in range(W)] for _ in range(H)]

        for col, c in enumerate(candles):
            bull        = c.close >= c.open
            body_style  = 'green' if bull else 'red'
            wick_style  = 'bright_black'

            r_high = to_row(c.high)
            r_low  = to_row(c.low)
            r_obod = to_row(max(c.open, c.close))  # body top (open/close higher)
            r_cbod = to_row(min(c.open, c.close))  # body bottom

            for row in range(r_high, r_low + 1):
                if r_obod <= row <= r_cbod:
                    grid[row][col] = ('|', body_style)
                else:
                    grid[row][col] = ('|', wick_style)

            # Overlay trade marker
            if c.ts in trade_map:
                tr = trade_map[c.ts]
                tr_row = to_row(tr.price)
                if tr.direction == 'LONG':
                    ch = '^'
                    if   tr.outcome == 'WIN':  sty = 'bold bright_green'
                    elif tr.outcome == 'LOSS': sty = 'bold bright_red'
                    else:                      sty = 'bright_green'
                else:
                    ch = 'v'
                    if   tr.outcome == 'WIN':  sty = 'bold bright_green'
                    elif tr.outcome == 'LOSS': sty = 'bold bright_red'
                    else:                      sty = 'bright_red'
                grid[tr_row][col] = (ch, sty)

        # Y-axis: show price at top, middle, bottom
        label_rows = {
            0:       f" ${p_max:,.0f}",
            H // 2:  f" ${(p_max + p_min) / 2:,.0f}",
            H - 1:   f" ${p_min:,.0f}",
        }

        t = Text()
        for row in range(H):
            t.append("  ")
            for col in range(W):
                ch, sty = grid[row][col]
                if sty:
                    t.append(ch, style=sty)
                else:
                    t.append(ch)
            label = label_rows.get(row, '')
            t.append(label, style='dim')
            t.append('\n')

        # Time axis
        if len(candles) >= 2:
            t_first = datetime.fromtimestamp(candles[0].ts / 1000, tz=timezone.utc).strftime('%H:%M')
            t_last  = datetime.fromtimestamp(candles[-1].ts / 1000, tz=timezone.utc).strftime('%H:%M')
            gap = max(1, W - len(t_first) - len(t_last) - 4)
            t.append(f"  {t_first}", style='dim')
            t.append(' ' * gap)
            t.append(f"{t_last} UTC\n", style='dim')

        # Legend
        t.append("  ", style='dim')
        t.append("^", style='bright_green')
        t.append(" LONG  ", style='dim')
        t.append("v", style='bright_red')
        t.append(" SHORT  ", style='dim')
        t.append("(bold = closed, green=WIN red=LOSS)", style='dim')

        return t


class Dashboard:
    """
    Builds and refreshes a Rich Live layout with all trading information.
    Designed to run in a background thread or be polled in the event loop.
    """

    def __init__(
        self,
        mode:        str,
        risk_mgr:    "RiskManager",
        atr:         "ATRIndicator",
        dv:          "DeltaVolumeIndicator",
        vwap:        "VWAPIndicator",
        cvd:         "CVDIndicator",
        vp:          "VolumeProfile",
        sr:          "SupportResistance",
        filters:     "TradeFilters",
        calendar:    "EconomicCalendar",
        nn_agent:    Optional["DQNAgent"]      = None,
        param_tuner: Optional["ParameterTuner"] = None,
    ):
        self.mode      = mode.upper()
        self._risk     = risk_mgr
        self._atr      = atr
        self._dv       = dv
        self._vwap     = vwap
        self._cvd      = cvd
        self._vp       = vp
        self._sr       = sr
        self._filters  = filters
        self._calendar = calendar
        self._nn        = nn_agent
        self._tuner     = param_tuner

        self._last_price:    float = 0.0
        self._last_signal:   Optional["Signal"] = None
        self._signal_log:    Deque[str] = deque(maxlen=6)
        self._live:          Optional[Live] = None
        self.chart:          CandleChart = CandleChart()

        # Position data injected externally
        self.open_position_str:  str   = "No open position"
        self.unrealized_pnl:     float = 0.0
        self.sim_stats:          dict  = {}

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> "Live":
        self._live = Live(
            self._render(),
            refresh_per_second=2,
            screen=True,
        )
        return self._live

    def update_price(self, price: float) -> None:
        self._last_price = price

    def log_signal(self, signal: "Signal") -> None:
        self._last_signal = signal
        ts = datetime.fromtimestamp(signal.timestamp).strftime("%H:%M:%S")
        conf = _CONF_CHARS.get(signal.confidence, "░░░")
        self._signal_log.appendleft(
            f"[dim]{ts}[/dim]  [{_dir_color(signal.direction)}]{signal.direction}[/]"
            f" @{signal.entry_price:,.2f}  [{conf}] — {signal.strategy}"
        )

    def log_close(self, outcome: str, pnl: float) -> None:
        if self._signal_log:
            last = self._signal_log[0]
            color = "green" if pnl >= 0 else "red"
            self._signal_log[0] = last + f"  [{color}]→ {outcome} {pnl:+.2f}[/]"

    def chart_trade_open(self, candle_ts: int, direction: str, price: float) -> None:
        self.chart.open_trade(candle_ts, direction, price)

    def chart_trade_close(self, outcome: str) -> None:
        self.chart.close_trade(outcome)

    def refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    # ── Rendering ───────────────────────────────────────────────────────────

    def _render(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(name="top",    size=12),
            Layout(name="stats",  size=3),
            Layout(name="filter", size=3),
            Layout(name="nn",     size=3),
            Layout(name="news",   size=2),
            Layout(name="log",    size=7),
            Layout(name="chart",  size=16),
        )
        layout["top"].split_row(
            Layout(name="price",      ratio=1),
            Layout(name="indicators", ratio=2),
            Layout(name="position",   ratio=2),
        )

        layout["price"].update(self._price_panel())
        layout["indicators"].update(self._indicators_panel())
        layout["position"].update(self._position_panel())
        layout["stats"].update(self._stats_panel())
        layout["filter"].update(self._filter_panel())
        layout["nn"].update(self._nn_panel())
        layout["news"].update(self._news_panel())
        layout["log"].update(self._log_panel())
        layout["chart"].update(self._chart_panel())

        mode_color = {"SIMULATION": "cyan", "ALERT": "yellow", "AUTO": "red"}.get(self.mode, "white")
        return Panel(
            layout,
            title=f"[bold white]BTCUSDT FUTURES  15s  MODE: "
                  f"[{mode_color}]{self.mode}[/{mode_color}][/bold white]",
            border_style="bright_blue",
        )

    def _price_panel(self) -> Panel:
        p  = self._last_price
        dv = self._dv
        t  = Text()

        price_color = "bright_green" if dv.last_delta >= 0 else "bright_red"
        t.append(f"\n  ${p:,.2f}\n", style=f"bold {price_color}")

        delta_color = "green" if dv.last_delta >= 0 else "red"
        sign = "+" if dv.last_delta >= 0 else ""
        t.append(f"  d  {sign}{dv.last_delta:,.2f}\n", style=delta_color)

        vol_x = dv.last_volume / dv.avg_volume if dv.avg_volume > 0 else 0
        vol_color = "yellow" if vol_x > 1.5 else "white"
        t.append(f"  Vol  {vol_x:.1f}× avg\n", style=vol_color)

        if dv.is_any_bubble:
            bdir  = dv.bubble_direction or ""
            bcol  = "bright_green" if bdir == "BUY" else "bright_red"
            t.append(f"\n  BUBBLE [{bdir}]\n", style=f"bold {bcol}")
            t.append(f"  {dv.bubble_strength:.1f}× avg\n", style=bcol)

        cvd_color = "green" if self._cvd.cvd >= 0 else "red"
        t.append(f"\n  CVD {self._cvd.cvd:+,.0f}\n", style=cvd_color)

        zone = self._vwap.zone
        zone_color = {
            "OB": "red", "STRONG_OB": "bright_red", "EXTREME_OB": "bold bright_red",
            "OS": "green", "STRONG_OS": "bright_green", "EXTREME_OS": "bold bright_green",
        }.get(zone, "white")
        t.append(f"  Zone  {zone}\n", style=zone_color)

        return Panel(t, title="[bold]PRICE[/bold]", border_style="blue")

    def _indicators_panel(self) -> Panel:
        vwap = self._vwap
        atr  = self._atr
        vp   = self._vp

        tbl = Table(box=None, show_header=False, padding=(0, 1))
        tbl.add_column(style="dim")
        tbl.add_column(style="bold white")

        if vwap.vwap > 0:
            dist = vwap.distance_to_vwap_pct
            sign = "+" if dist >= 0 else ""
            tbl.add_row("VWAP",    f"${vwap.vwap:,.2f}  ({sign}{dist:.2f}%)")
            tbl.add_row("Upper1",  f"${vwap.upper1:,.2f}")
            tbl.add_row("Lower1",  f"${vwap.lower1:,.2f}")
            tbl.add_row("HTF",     f"${vwap.htf_vwap:,.2f}  [{vwap.htf_trend}]")
        else:
            tbl.add_row("VWAP", "warming up…")

        tbl.add_row("", "")

        if atr.is_ready:
            tbl.add_row("ATR",  f"${atr.atr:,.2f}  ({atr.atr_pct:.3f}%)  [{atr.volatility_state}]")
        else:
            tbl.add_row("ATR", "warming up…")

        tbl.add_row("", "")

        if vp.poc > 0:
            tbl.add_row("POC", f"${vp.poc:,.2f}")
            tbl.add_row("VAH", f"${vp.vah:,.2f}")
            tbl.add_row("VAL", f"${vp.val:,.2f}")
        else:
            tbl.add_row("VP",  "warming up…")

        # Absorption / Exhaustion
        tbl.add_row("", "")
        if self._cvd.absorption_detected:
            tbl.add_row("[yellow]ABSORPTION[/yellow]", self._cvd.absorption_direction)
        if self._cvd.exhaustion_detected:
            tbl.add_row("[magenta]EXHAUSTION[/magenta]", self._cvd.exhaustion_direction)

        return Panel(tbl, title="[bold]INDICATORS[/bold]", border_style="blue")

    def _position_panel(self) -> Panel:
        t = Text()
        sig = self._last_signal

        if self.open_position_str and self.open_position_str != "No open position":
            t.append(f"\n  {self.open_position_str}\n", style="bold yellow")
            upnl_color = "green" if self.unrealized_pnl >= 0 else "red"
            t.append(f"  uPnL: {self.unrealized_pnl:+.2f} USDT\n", style=upnl_color)
        else:
            t.append("\n  No open position\n", style="dim")

        if sig:
            t.append("\n  Last Signal:\n", style="bold")
            dc = _dir_color(sig.direction)
            t.append(f"  [{sig.direction}] {sig.strategy}\n", style=f"bold {dc}")
            t.append(f"  Entry:  ${sig.entry_price:,.2f}\n", style="white")
            t.append(f"  SL:     ${sig.stop_loss:,.2f}\n",   style="red")
            t.append(f"  TP:     ${sig.take_profit:,.2f}\n", style="green")
            t.append(f"  R:R     1:{sig.rr_ratio:.1f}\n",    style="yellow")
            conf_bar = _CONF_CHARS.get(sig.confidence, "░░░")
            t.append(f"  Conf:   [{conf_bar}] {sig.confidence}/3\n", style="cyan")

        return Panel(t, title="[bold]POSITION / SIGNAL[/bold]", border_style="blue")

    def _stats_panel(self) -> Panel:
        s    = self.sim_stats
        bal  = s.get("balance", self._risk.balance)
        pnl  = s.get("session_pnl", 0.0)
        w    = s.get("wins",   self._risk.session_wins)
        l    = s.get("losses", self._risk.session_losses)
        tot  = s.get("total_trades", self._risk.total_trades)
        rem  = s.get("losses_left",  self._risk.losses_remaining())
        wr   = s.get("win_rate",     self._risk.win_rate)
        dd   = s.get("max_drawdown", 0.0)
        fees = s.get("fees_paid", 0.0)

        pnl_color    = "green" if pnl >= 0 else "red"
        losses_color = "yellow" if rem > 1 else "red"

        t = Text()
        t.append(f"  SESSION: {tot} trades  │  ", style="dim")
        t.append(f"{w}W  {l}L  ", style="white")
        t.append(f"({wr:.0%})  │  ", style="dim")
        t.append(f"PnL: {pnl:+.2f} USDT  │  ", style=pnl_color)
        t.append(f"Fees: -{fees:.4f}  │  ", style="yellow")
        t.append(f"Balance: ${bal:,.2f}  │  ", style="bold white")
        if rem == 999:
            t.append("Losses: no limit", style="dim")
        else:
            t.append(f"Losses left: {rem}/{config.MAX_LOSSES}", style=losses_color)
        if dd > 0:
            t.append(f"  │  MaxDD: {dd:.1%}", style="red")

        return Panel(t, title="[bold]SESSION STATS[/bold]", border_style="blue",
                     padding=(0, 1))

    def _filter_panel(self) -> Panel:
        fstat = self._filters.status_dict()
        items = [
            ("Vol",       fstat.get("volume_ok", True)),
            ("News",      fstat.get("news_ok", True)),
            ("Volatility",fstat.get("volatility_ok", True)),
        ]
        t = Text()
        t.append("  FILTERS: ", style="dim")
        for name, ok in items:
            icon  = "OK" if ok else "!!"
            color = "green" if ok else "red"
            t.append(f"{name} [{icon}]  ", style=color)
        return Panel(t, box=box.SIMPLE, padding=(0, 0))

    def _news_panel(self) -> Panel:
        try:
            nxt = self._calendar.next_high_impact()
            if nxt:
                mins = nxt.minutes_until
                if mins < 0:
                    time_str = f"{abs(mins):.0f}m ago"
                    color = "yellow"
                else:
                    h = int(mins // 60)
                    m = int(mins % 60)
                    time_str = f"in {h}h {m}m" if h > 0 else f"in {m}m"
                    color = "red" if mins < config.NEWS_BUFFER_MIN else "yellow"
                t = Text()
                t.append(f"  NEWS: {nxt.name}  {nxt.dt.strftime('%H:%M')} UTC  ({time_str})  [!] HIGH", style=color)
            else:
                t = Text("  No high-impact news scheduled", style="dim")
        except Exception:
            t = Text("  Economic calendar unavailable", style="dim")
        return Panel(t, box=box.SIMPLE, padding=(0, 0))

    def _nn_panel(self) -> Panel:
        if self._nn is None:
            t = Text("  NN agent not initialised", style="dim")
            return Panel(t, box=box.SIMPLE, padding=(0, 0))

        s = self._nn.stats()
        eps      = s.get("epsilon", 0.0)
        eps_pct  = s.get("eps_pct", 0.0)
        eps_flr  = s.get("eps_floor", 0.05)
        steps    = s.get("steps", 0)
        tot_r    = s.get("total_reward", 0.0)
        loss     = s.get("last_loss")
        recent   = s.get("recent", "-")
        seq_len  = s.get("seq_len", "?")
        buf_sz   = s.get("buf_size", 0)
        buf_cap  = s.get("buf_capacity", 1)
        buf_pct  = s.get("buf_pct", 0.0)
        lr       = s.get("lr", 0.0)
        sdim     = s.get("state_dim", "?")

        loss_str  = f"{loss:.4f}" if loss is not None else "n/a"
        eps_color = "green" if eps < 0.3 else ("yellow" if eps < 0.6 else "red")
        r_color   = "green" if tot_r >= 0 else "red"
        buf_color = "green" if buf_pct > 80 else ("yellow" if buf_pct > 30 else "red")

        t = Text()
        # Line 1: runtime stats
        t.append("  NN: ", style="dim")
        t.append(f"e={eps:.3f}", style=eps_color)
        t.append(f"({eps_pct:.0f}%->floor={eps_flr})  ", style="dim")
        t.append(f"Loss:{loss_str}  ", style="white")
        t.append(f"TotalR:", style="dim")
        t.append(f"{tot_r:+.1f}  ", style=r_color)
        t.append(f"Recent:[{recent}]  ", style="cyan")

        # Line 2: config + buffer
        t.append("\n  Cfg: ", style="dim")
        t.append(f"LR={lr}  ", style="white")
        t.append(f"dim={sdim}  ", style="white")
        t.append(f"Seq:{seq_len}  ", style="white")
        t.append(f"Steps:{steps}  ", style="dim")
        t.append(f"Buf:", style="dim")
        t.append(f"{buf_sz:,}/{buf_cap:,} ({buf_pct:.0f}%)", style=buf_color)
        t.append("  [LSTM]", style="bright_cyan")

        # Second line: tuned parameter values vs defaults
        _ABBREV = {
            "DELTA_BUBBLE_MULT":  ("BubM",  2.0,   False),
            "ABSORPTION_BODY":    ("AbsB",  0.30,  False),
            "LOW_VOLUME_FACTOR":  ("LVol",  0.50,  False),
            "IMBALANCE_RATIO":    ("Imb",   3.0,   False),
            "SWEEP_REVERSAL_PCT": ("Swp",   0.03,  False),
            "NN_SEQ_LEN":         ("SeqL",  10.0,  True),   # shown as int
        }
        if self._tuner is not None:
            vals = self._tuner.current_values()
            t.append("\n  Params: ", style="dim")
            for key, (short, default, as_int) in _ABBREV.items():
                val = vals.get(key, default)
                drift_pct = abs(val - default) / default if default else 0.0
                color = "white" if drift_pct < 0.02 else (
                    "yellow" if drift_pct < 0.10 else "magenta"
                )
                t.append(f"{short}:", style="dim")
                disp = f"{int(round(val))}" if as_int else f"{val:.3f}"
                t.append(f"{disp}  ", style=color)
        else:
            t.append("\n  Params: tuner not active", style="dim")

        return Panel(t, box=box.SIMPLE, padding=(0, 0))

    def _log_panel(self) -> Panel:
        t = Text()
        for line in self._signal_log:
            t.append(f"  {line}\n")
        if not self._signal_log:
            t.append("  Waiting for signals...", style="dim")
        return Panel(t, title="[bold]SIGNAL LOG[/bold]", border_style="blue")

    def _chart_panel(self) -> Panel:
        # Estimate usable content width: Rich panels have ~2 char borders each side
        # We pass a conservative fixed width; the chart clips to available candles.
        chart_text = self.chart.render(width=110)
        return Panel(
            chart_text,
            title="[bold]15s CHART[/bold]",
            border_style="blue",
        )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _dir_color(direction: str) -> str:
    return "bright_green" if direction == "LONG" else "bright_red"
