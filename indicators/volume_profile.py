"""
volume_profile.py — Fixed Range Volume Profile.

Builds a volume histogram over the last N candles to identify:
  POC  — Point of Control (highest volume price level)
  VAH  — Value Area High  (upper edge of 70% volume zone)
  VAL  — Value Area Low   (lower edge of 70% volume zone)

These levels act as dynamic support/resistance.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

import config
from data.market_data import Candle


class VolumeProfile:
    """
    Rolling fixed-range volume profile over the last `window` candles.
    Recalculates on every new candle.
    """

    def __init__(self, window: int = 100, bins: int = config.VP_BINS):
        self.window = window
        self.bins   = bins
        self._candles: Deque[Candle] = deque(maxlen=window)

        # Computed levels
        self.poc: float = 0.0
        self.vah: float = 0.0
        self.val: float = 0.0

        # Full profile: list of (price_mid, volume) sorted by price
        self.profile: List[Tuple[float, float]] = []

        # High-volume nodes (volume > 1.5× average bin volume)
        self.hvn_levels: List[float] = []
        # Low-volume nodes (volume < 0.5× average bin volume)
        self.lvn_levels: List[float] = []

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        self._candles.append(candle)
        if len(self._candles) >= 10:
            self._calculate()

    # ── Queries ─────────────────────────────────────────────────────────────

    def is_in_value_area(self, price: float) -> bool:
        return self.val <= price <= self.vah

    def nearest_level(self, price: float) -> Optional[float]:
        """Return the nearest S/R level from POC, VAH, VAL."""
        levels = [l for l in [self.poc, self.vah, self.val] if l > 0]
        if not levels:
            return None
        return min(levels, key=lambda l: abs(l - price))

    def distance_to_poc_pct(self, price: float) -> float:
        if self.poc == 0:
            return 0.0
        return (price - self.poc) / self.poc * 100.0

    def get_support_resistance(self) -> Dict[str, List[float]]:
        """Return dict with 'support' and 'resistance' level lists."""
        return {
            "support":    [l for l in self.hvn_levels if l < self.poc],
            "resistance": [l for l in self.hvn_levels if l > self.poc],
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _calculate(self) -> None:
        candles = list(self._candles)
        if not candles:
            return

        # Price range
        low  = min(c.low  for c in candles)
        high = max(c.high for c in candles)
        if high <= low:
            return

        bin_size = (high - low) / self.bins
        if bin_size == 0:
            return

        # Distribute each candle's volume across its price range
        vol_bins = np.zeros(self.bins)
        for c in candles:
            lo_bin = max(0, int((c.low  - low) / bin_size))
            hi_bin = min(self.bins - 1, int((c.high - low) / bin_size))
            n_bins = hi_bin - lo_bin + 1
            if n_bins <= 0:
                continue
            vol_per_bin = c.volume / n_bins
            vol_bins[lo_bin : hi_bin + 1] += vol_per_bin

        total_vol = vol_bins.sum()
        if total_vol == 0:
            return

        # POC
        poc_idx   = int(np.argmax(vol_bins))
        self.poc  = low + (poc_idx + 0.5) * bin_size

        # Value Area (70% of volume, expanding from POC outward)
        target     = total_vol * config.VP_VALUE_AREA_PCT
        accumulated = vol_bins[poc_idx]
        lo_idx = hi_idx = poc_idx

        while accumulated < target:
            can_go_lo = lo_idx > 0
            can_go_hi = hi_idx < self.bins - 1
            if not can_go_lo and not can_go_hi:
                break
            add_lo = vol_bins[lo_idx - 1] if can_go_lo else -1
            add_hi = vol_bins[hi_idx + 1] if can_go_hi else -1
            if add_lo >= add_hi:
                lo_idx  -= 1
                accumulated += vol_bins[lo_idx]
            else:
                hi_idx  += 1
                accumulated += vol_bins[hi_idx]

        self.val = low + lo_idx * bin_size
        self.vah = low + (hi_idx + 1) * bin_size

        # Build full profile list
        avg_bin_vol = total_vol / self.bins
        self.profile    = []
        self.hvn_levels = []
        self.lvn_levels = []
        for i, v in enumerate(vol_bins):
            price_mid = low + (i + 0.5) * bin_size
            self.profile.append((price_mid, v))
            if v > avg_bin_vol * 1.5:
                self.hvn_levels.append(price_mid)
            elif v < avg_bin_vol * 0.5:
                self.lvn_levels.append(price_mid)
