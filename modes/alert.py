"""
modes/alert.py — Signal alert mode.

When a signal fires, this mode:
  1. Flashes a bold colored Rich panel to the terminal
  2. Plays a Windows system beep sequence (winsound)
  3. Prints full trade details: entry, SL, TP, R:R, strategy, reason

No orders are placed.  The user decides whether to enter manually.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import config
from data.market_data import Signal

logger = logging.getLogger(__name__)

console = Console()


class AlertMode:
    """
    Receives Signal objects and notifies the user via terminal + sound.
    """

    def __init__(self):
        self._alert_log: List[Signal] = []
        self._on_alert_callbacks: List[Callable[[Signal], None]] = []

    def add_callback(self, cb: Callable[[Signal], None]) -> None:
        self._on_alert_callbacks.append(cb)

    # ── Process signal ───────────────────────────────────────────────────────

    def process_signal(self, signal: Signal) -> None:
        """
        Called when a new signal is generated.
        Plays sound and prints visual alert.
        """
        self._alert_log.append(signal)
        self._print_alert(signal)
        self._play_sound(signal.direction)
        for cb in self._on_alert_callbacks:
            try:
                cb(signal)
            except Exception:
                pass

    # ── Visual alert ─────────────────────────────────────────────────────────

    def _print_alert(self, signal: Signal) -> None:
        color = "bright_green" if signal.direction == "LONG" else "bright_red"
        conf_bar = "#" * signal.confidence + "-" * (3 - signal.confidence)

        from datetime import datetime
        ts = datetime.fromtimestamp(signal.timestamp).strftime("%H:%M:%S")

        text = Text()
        arrow = "^" if signal.direction == "LONG" else "v"
        text.append(f"\n  {arrow}  {signal.direction}  --  {signal.strategy}\n", style=f"bold {color}")
        text.append(f"\n  Entry:       ", style="bold white")
        text.append(f"${signal.entry_price:,.2f}\n", style=f"bold {color}")
        text.append(f"  Stop Loss:   ", style="bold white")
        text.append(f"${signal.stop_loss:,.2f}\n", style="bold red")
        text.append(f"  Take Profit: ", style="bold white")
        text.append(f"${signal.take_profit:,.2f}\n", style="bold green")
        text.append(f"  R:R Ratio:   ", style="bold white")
        text.append(f"1:{signal.rr_ratio:.1f}\n", style="bold yellow")
        text.append(f"  Confidence:  ", style="bold white")
        text.append(f"[{conf_bar}] {signal.confidence}/3\n", style="bold cyan")
        text.append(f"  Reason:      ", style="bold white")
        text.append(f"{signal.reason}\n", style="dim")
        text.append(f"\n  Time: {ts}", style="dim")

        panel = Panel(
            text,
            title=f"[bold {color}]  SIGNAL: {signal.direction}  [/bold {color}]",
            border_style=color,
            expand=False,
            padding=(0, 2),
        )
        console.print()
        console.print(panel)
        console.print()

    # ── Sound alert ──────────────────────────────────────────────────────────

    def _play_sound(self, direction: str) -> None:
        if not config.ALERT_BEEP_COUNT:
            return
        thread = threading.Thread(
            target=self._beep_sequence,
            args=(direction,),
            daemon=True,
        )
        thread.start()

    def _beep_sequence(self, direction: str) -> None:
        try:
            import winsound
            count = config.ALERT_BEEP_COUNT
            if direction == "LONG":
                # Ascending tones for LONG
                for freq in [800, 1000, 1200]:
                    winsound.Beep(freq, 150)
                    time.sleep(0.05)
            else:
                # Descending tones for SHORT
                for freq in [1200, 1000, 800]:
                    winsound.Beep(freq, 150)
                    time.sleep(0.05)
            # Additional beeps
            for _ in range(count - 1):
                winsound.Beep(1000, 200)
                time.sleep(0.1)
        except ImportError:
            # Non-Windows fallback
            print("\a" * config.ALERT_BEEP_COUNT)

    # ── Log ─────────────────────────────────────────────────────────────────

    @property
    def alert_history(self) -> List[Signal]:
        return list(self._alert_log)

    def last_alert(self) -> Optional[Signal]:
        return self._alert_log[-1] if self._alert_log else None
