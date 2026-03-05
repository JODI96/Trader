"""
delta_volume.py — Delta Volume tracking and bubble detection.

Implements the logic described in the "Delta Volume Bubbles" TradingView script:
  - Track buy/sell volume delta per candle
  - A "bubble" fires when |delta| exceeds DELTA_BUBBLE_MULT × rolling average
  - Bubble direction: positive delta bubble = BUY pressure, negative = SELL pressure
  - Also tracks cumulative session delta (resets daily)
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

import config
from data.market_data import Candle


class DeltaVolumeIndicator:
    """
    Keeps a rolling window of per-candle delta values and detects
    "volume bubbles" — abnormally large directional volume spikes.
    """

    WINDOW = 10   # rolling window for average delta calculation (10 × 1m = 10-min baseline)

    def __init__(self):
        self._deltas:     Deque[float] = deque(maxlen=self.WINDOW)
        self._abs_deltas: Deque[float] = deque(maxlen=self.WINDOW)
        self._volumes:    Deque[float] = deque(maxlen=self.WINDOW)

        # Latest candle state
        self.last_delta:    float = 0.0
        self.last_buy_vol:  float = 0.0
        self.last_sell_vol: float = 0.0
        self.last_volume:   float = 0.0

        # Session cumulative delta (reset at 00:00 UTC)
        self.session_cum_delta: float = 0.0
        self._last_session_day: int   = -1

        # Bubble state
        self.is_buy_bubble:  bool  = False
        self.is_sell_bubble: bool  = False
        self.bubble_strength: float = 0.0   # multiple of average

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        self._reset_session_if_needed(candle)

        self.last_delta    = candle.delta
        self.last_buy_vol  = candle.buy_vol
        self.last_sell_vol = candle.sell_vol
        self.last_volume   = candle.volume

        self._deltas.append(candle.delta)
        self._abs_deltas.append(abs(candle.delta))
        self._volumes.append(candle.volume)

        self.session_cum_delta += candle.delta

        self._detect_bubble()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def avg_abs_delta(self) -> float:
        if not self._abs_deltas:
            return 0.0
        return float(np.mean(self._abs_deltas))

    @property
    def avg_volume(self) -> float:
        if not self._volumes:
            return 0.0
        return float(np.mean(self._volumes))

    @property
    def buy_sell_ratio(self) -> float:
        """buy_vol / total_vol for the last candle (0.5 = balanced)."""
        if self.last_volume == 0:
            return 0.5
        return self.last_buy_vol / self.last_volume

    @property
    def is_any_bubble(self) -> bool:
        return self.is_buy_bubble or self.is_sell_bubble

    @property
    def bubble_direction(self) -> Optional[str]:
        if self.is_buy_bubble:
            return "BUY"
        if self.is_sell_bubble:
            return "SELL"
        return None

    @property
    def delta_trend(self) -> str:
        """Short-term delta momentum: 'BULLISH', 'BEARISH', or 'NEUTRAL'."""
        if len(self._deltas) < 3:
            return "NEUTRAL"
        recent = list(self._deltas)[-5:]
        total  = sum(recent)
        if total > self.avg_abs_delta * 0.5:
            return "BULLISH"
        if total < -self.avg_abs_delta * 0.5:
            return "BEARISH"
        return "NEUTRAL"

    # ── Internal ────────────────────────────────────────────────────────────

    def _detect_bubble(self) -> None:
        avg = self.avg_abs_delta
        if avg == 0:
            self.is_buy_bubble  = False
            self.is_sell_bubble = False
            self.bubble_strength = 0.0
            return

        strength = abs(self.last_delta) / avg
        self.bubble_strength = strength

        if strength >= config.DELTA_BUBBLE_MULT:
            self.is_buy_bubble  = self.last_delta > 0
            self.is_sell_bubble = self.last_delta < 0
        else:
            self.is_buy_bubble  = False
            self.is_sell_bubble = False

    def _reset_session_if_needed(self, candle: Candle) -> None:
        from datetime import datetime, timezone
        day = datetime.fromtimestamp(candle.ts / 1000, tz=timezone.utc).day
        if day != self._last_session_day:
            self.session_cum_delta   = 0.0
            self._last_session_day   = day
