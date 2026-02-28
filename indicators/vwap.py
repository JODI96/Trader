"""
vwap.py — Session VWAP with standard-deviation bands.

Implements the "Hoss VWAP AD" concept:
  - VWAP resets every day at 00:00 UTC
  - Bands at ±1σ, ±2σ, ±3σ derived from volume-weighted variance
  - Overbought:  price > VWAP + 1σ
  - Oversold:    price < VWAP - 1σ
  - Extreme OB:  price > VWAP + 2σ
  - Extreme OS:  price < VWAP - 2σ

Also provides HTF VWAP (15-min candles) for trend bias.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import config
from data.market_data import Candle


class VWAPIndicator:
    """
    Session VWAP with ±1/2/3σ bands.
    Call `update(candle)` on each new scalp (15s) candle.
    Call `update_htf(candle)` on each new HTF (15min) candle.
    """

    def __init__(self):
        # Session accumulators (reset daily)
        self._sum_tv: float = 0.0    # Σ(typical_price × volume)
        self._sum_v:  float = 0.0    # Σ(volume)
        self._sum_tv2: float = 0.0   # Σ(typical_price² × volume)  for variance
        self._session_day: int = -1

        # Computed values
        self.vwap:    float = 0.0
        self.sigma:   float = 0.0
        self.upper1:  float = 0.0
        self.upper2:  float = 0.0
        self.upper3:  float = 0.0
        self.lower1:  float = 0.0
        self.lower2:  float = 0.0
        self.lower3:  float = 0.0

        # HTF VWAP (separate session accumulators)
        self._htf_sum_tv:  float = 0.0
        self._htf_sum_v:   float = 0.0
        self._htf_sum_tv2: float = 0.0
        self._htf_day:     int   = -1
        self.htf_vwap:     float = 0.0

        self.last_price:   float = 0.0

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        self._reset_session_if_needed(candle)

        tp = candle.typical_price
        v  = candle.volume
        self._sum_tv  += tp * v
        self._sum_v   += v
        self._sum_tv2 += tp * tp * v
        self.last_price = candle.close

        self._recalculate()

    def update_htf(self, candle: Candle) -> None:
        from datetime import datetime, timezone
        day = datetime.fromtimestamp(candle.ts / 1000, tz=timezone.utc).day
        if day != self._htf_day:
            self._htf_sum_tv  = 0.0
            self._htf_sum_v   = 0.0
            self._htf_sum_tv2 = 0.0
            self._htf_day     = day

        tp = candle.typical_price
        v  = candle.volume
        self._htf_sum_tv  += tp * v
        self._htf_sum_v   += v
        if self._htf_sum_v > 0:
            self.htf_vwap = self._htf_sum_tv / self._htf_sum_v

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def zone(self) -> str:
        """
        Returns the current VWAP zone based on price vs bands.
        Used by VWAP Bounce strategy and as overbought/oversold filter.
        """
        if self.vwap == 0:
            return "NEUTRAL"
        p = self.last_price
        if p >= self.upper3:
            return "EXTREME_OB"
        if p >= self.upper2:
            return "STRONG_OB"
        if p >= self.upper1:
            return "OB"           # overbought — look for shorts
        if p <= self.lower3:
            return "EXTREME_OS"
        if p <= self.lower2:
            return "STRONG_OS"
        if p <= self.lower1:
            return "OS"           # oversold — look for longs
        return "NEUTRAL"

    @property
    def distance_to_vwap_pct(self) -> float:
        """Price distance from VWAP as percentage."""
        if self.vwap == 0:
            return 0.0
        return (self.last_price - self.vwap) / self.vwap * 100.0

    @property
    def htf_trend(self) -> str:
        """Trend bias from 15-min VWAP: 'BULLISH', 'BEARISH', 'NEUTRAL'."""
        if self.htf_vwap == 0:
            return "NEUTRAL"
        if self.last_price > self.htf_vwap * 1.0002:
            return "BULLISH"
        if self.last_price < self.htf_vwap * 0.9998:
            return "BEARISH"
        return "NEUTRAL"

    def is_at_vwap(self, tolerance_sigma: float = 0.3) -> bool:
        """Price is within tolerance_sigma × sigma of VWAP."""
        if self.sigma == 0:
            return False
        return abs(self.last_price - self.vwap) <= tolerance_sigma * self.sigma

    def is_at_band(self, band: float = 1.0, tolerance_sigma: float = 0.2) -> str:
        """
        Check if price is near a VWAP band.
        Returns 'UPPER', 'LOWER', or '' (not at band).
        """
        if self.sigma == 0:
            return ""
        target_upper = self.vwap + band * self.sigma
        target_lower = self.vwap - band * self.sigma
        tol = tolerance_sigma * self.sigma
        if abs(self.last_price - target_upper) <= tol:
            return "UPPER"
        if abs(self.last_price - target_lower) <= tol:
            return "LOWER"
        return ""

    # ── Internal ────────────────────────────────────────────────────────────

    def _recalculate(self) -> None:
        if self._sum_v == 0:
            return
        self.vwap = self._sum_tv / self._sum_v
        # Volume-weighted variance: E[x²] - E[x]²
        variance  = max(0.0, self._sum_tv2 / self._sum_v - self.vwap ** 2)
        self.sigma = math.sqrt(variance)
        self.upper1 = self.vwap + 1.0 * self.sigma
        self.upper2 = self.vwap + 2.0 * self.sigma
        self.upper3 = self.vwap + 3.0 * self.sigma
        self.lower1 = self.vwap - 1.0 * self.sigma
        self.lower2 = self.vwap - 2.0 * self.sigma
        self.lower3 = self.vwap - 3.0 * self.sigma

    def _reset_session_if_needed(self, candle: Candle) -> None:
        from datetime import datetime, timezone
        day = datetime.fromtimestamp(candle.ts / 1000, tz=timezone.utc).day
        if day != self._session_day:
            self._sum_tv      = 0.0
            self._sum_v       = 0.0
            self._sum_tv2     = 0.0
            self._session_day = day
