"""
atr.py — Average True Range indicator.

Used for:
  - Stop-loss sizing (structure-based SL = ATR multiple from entry)
  - Volatility filter (minimum ATR required to trade)
  - Absorption detection threshold (body < ATR × factor)
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import config
from data.market_data import Candle


class ATRIndicator:
    """
    Exponential Moving Average of True Range (Wilder's ATR).
    alpha = 1 / period  (Wilder smoothing)
    """

    def __init__(self, period: int = config.ATR_PERIOD):
        self.period = period
        self._alpha = 1.0 / period
        self.atr:  float = 0.0
        self._prev_close: Optional[float] = None
        self._warmup_trs: Deque[float] = deque(maxlen=period)
        self._is_warm: bool = False

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        tr = self._true_range(candle)

        if not self._is_warm:
            self._warmup_trs.append(tr)
            if len(self._warmup_trs) == self.period:
                self.atr     = sum(self._warmup_trs) / self.period
                self._is_warm = True
        else:
            # Wilder smoothing: ATR = prev_ATR × (1 - α) + TR × α
            self.atr = self.atr * (1.0 - self._alpha) + tr * self._alpha

        self._prev_close = candle.close

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def atr_pct(self) -> float:
        """ATR as percentage of last close."""
        if self._prev_close and self._prev_close > 0:
            return self.atr / self._prev_close * 100.0
        return 0.0

    @property
    def is_ready(self) -> bool:
        return self._is_warm

    @property
    def volatility_state(self) -> str:
        """
        Classifies current volatility.
        Based on ATR % of price:
          < 0.05%  → LOW
          0.05–0.15% → NORMAL
          > 0.15%  → HIGH
        """
        pct = self.atr_pct
        if pct < 0.05:
            return "LOW"
        if pct > 0.15:
            return "HIGH"
        return "NORMAL"

    def sl_distance(self, multiplier: float = 1.5) -> float:
        """Suggested stop-loss distance = ATR × multiplier."""
        return self.atr * multiplier

    # ── Internal ────────────────────────────────────────────────────────────

    def _true_range(self, candle: Candle) -> float:
        if self._prev_close is None:
            return candle.high - candle.low
        return max(
            candle.high - candle.low,
            abs(candle.high - self._prev_close),
            abs(candle.low  - self._prev_close),
        )
