"""
support_resistance.py — Swing pivot detection and equal H/L liquidity pools.

Implements concepts from "Traders Reality Main" (Etj1ixAs):
  - Swing highs and lows from price action (3-bar pivot)
  - Equal highs / equal lows detection (liquidity pools for sweep strategies)
  - Levels weighted by number of touches and ATR distance

Also uses HTF (15-min) candles to find larger S/R zones.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import config
from data.market_data import Candle


@dataclass
class SRLevel:
    price: float
    touches: int = 1
    is_high: bool = True    # True = resistance origin, False = support origin
    last_ts: int = 0

    @property
    def is_resistance(self) -> bool:
        return self.is_high

    @property
    def is_support(self) -> bool:
        return not self.is_high


class SupportResistance:
    """
    Detects swing pivots and equal H/L from price history.
    Maintains separate lists for scalp (15s) and HTF (15min) levels.
    """

    PIVOT_BARS = 3          # look-back bars each side for swing detection
    MAX_LEVELS = 20         # max S/R levels to keep

    def __init__(self):
        self._scalp_history: Deque[Candle] = deque(maxlen=config.LOOKBACK_CANDLES)
        self._htf_history:   Deque[Candle] = deque(maxlen=config.HTF_LOOKBACK)

        self.support_levels:    List[SRLevel] = []
        self.resistance_levels: List[SRLevel] = []

        # Equal highs / lows (liquidity pools)
        self.equal_highs: List[float] = []
        self.equal_lows:  List[float] = []

        # HTF levels (larger S/R zones)
        self.htf_support:    List[SRLevel] = []
        self.htf_resistance: List[SRLevel] = []

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        self._scalp_history.append(candle)
        if len(self._scalp_history) >= self.PIVOT_BARS * 2 + 1:
            self._detect_pivots(list(self._scalp_history),
                                self.support_levels, self.resistance_levels)
            self._detect_equal_hl(list(self._scalp_history))
            self._prune(self.support_levels)
            self._prune(self.resistance_levels)

    def update_htf(self, candle: Candle) -> None:
        self._htf_history.append(candle)
        if len(self._htf_history) >= self.PIVOT_BARS * 2 + 1:
            self._detect_pivots(list(self._htf_history),
                                self.htf_support, self.htf_resistance)
            self._prune(self.htf_support)
            self._prune(self.htf_resistance)

    # ── Queries ─────────────────────────────────────────────────────────────

    def nearest_support(self, price: float) -> Optional[float]:
        below = [l.price for l in self.support_levels if l.price < price]
        if not below:
            below = [l.price for l in self.htf_support if l.price < price]
        return max(below) if below else None

    def nearest_resistance(self, price: float) -> Optional[float]:
        above = [l.price for l in self.resistance_levels if l.price > price]
        if not above:
            above = [l.price for l in self.htf_resistance if l.price > price]
        return min(above) if above else None

    def is_near_level(self, price: float, tolerance_pct: float = 0.05) -> bool:
        """Return True if price is within tolerance_pct of any known S/R."""
        all_levels = (
            [l.price for l in self.support_levels] +
            [l.price for l in self.resistance_levels] +
            [l.price for l in self.htf_support] +
            [l.price for l in self.htf_resistance]
        )
        tol = price * tolerance_pct / 100.0
        return any(abs(price - l) <= tol for l in all_levels)

    def swept_equal_high(self, candle: Candle) -> Optional[float]:
        """
        If candle wicked above an equal high and closed back below it,
        return the swept level (stop hunt detected).
        """
        for level in self.equal_highs:
            if candle.high > level and candle.close < level:
                return level
        return None

    def swept_equal_low(self, candle: Candle) -> Optional[float]:
        """
        If candle wicked below an equal low and closed back above it,
        return the swept level.
        """
        for level in self.equal_lows:
            if candle.low < level and candle.close > level:
                return level
        return None

    # ── Internal ────────────────────────────────────────────────────────────

    def _detect_pivots(self, candles: List[Candle],
                       supports: List[SRLevel],
                       resistances: List[SRLevel]) -> None:
        n = len(candles)
        lb = self.PIVOT_BARS
        if n < lb * 2 + 1:
            return
        # Check the candle that is `lb` bars before the last
        idx = n - lb - 1
        c   = candles[idx]

        # Swing high: highest high in the window
        window_highs = [candles[i].high for i in range(idx - lb, idx + lb + 1)]
        window_lows  = [candles[i].low  for i in range(idx - lb, idx + lb + 1)]

        if c.high == max(window_highs):
            self._add_or_touch(resistances, c.high, is_high=True, ts=c.ts)

        if c.low == min(window_lows):
            self._add_or_touch(supports, c.low, is_high=False, ts=c.ts)

    def _detect_equal_hl(self, candles: List[Candle]) -> None:
        """
        Find price levels where 2+ swing highs or lows are within
        EQUAL_HL_TOL_PCT of each other — these are liquidity pools.
        """
        tol = config.EQUAL_HL_TOL_PCT / 100.0

        swing_highs = [l.price for l in self.resistance_levels]
        swing_lows  = [l.price for l in self.support_levels]

        equal_highs: List[float] = []
        for i, h1 in enumerate(swing_highs):
            for h2 in swing_highs[i+1:]:
                if abs(h1 - h2) / h1 <= tol:
                    equal_highs.append((h1 + h2) / 2.0)
        self.equal_highs = list(set(round(x, 2) for x in equal_highs))[:10]

        equal_lows: List[float] = []
        for i, l1 in enumerate(swing_lows):
            for l2 in swing_lows[i+1:]:
                if l1 > 0 and abs(l1 - l2) / l1 <= tol:
                    equal_lows.append((l1 + l2) / 2.0)
        self.equal_lows = list(set(round(x, 2) for x in equal_lows))[:10]

    def _add_or_touch(self, levels: List[SRLevel], price: float,
                      is_high: bool, ts: int) -> None:
        tol = price * config.EQUAL_HL_TOL_PCT / 100.0
        for lvl in levels:
            if abs(lvl.price - price) <= tol:
                lvl.touches += 1
                lvl.last_ts  = ts
                # Update price to average
                lvl.price = (lvl.price * (lvl.touches - 1) + price) / lvl.touches
                return
        levels.append(SRLevel(price=price, is_high=is_high, last_ts=ts))

    def _prune(self, levels: List[SRLevel]) -> None:
        if len(levels) > self.MAX_LEVELS:
            # Keep most-touched and most-recent
            levels.sort(key=lambda l: (l.touches, l.last_ts), reverse=True)
            del levels[self.MAX_LEVELS:]
