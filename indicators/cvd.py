"""
cvd.py — Cumulative Volume Delta (CVD) candles.

Implements concepts from:
  - "CVD Cumulative Volume Delta Candles" (NlM312nK)
  - "Cumulative Delta Volume" (vB1T3EMp)

Key signals:
  Absorption — high volume candle but CVD barely moves
               → passive orders absorbing aggressive flow at the level
  Exhaustion  — CVD diverges significantly from price direction
               → aggressive flow is running out of steam

CVD resets at the start of each session (00:00 UTC).
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

import config
from data.market_data import Candle


class CVDIndicator:
    """
    Tracks Cumulative Volume Delta and detects absorption / exhaustion patterns.
    """

    LOOKBACK = 10  # candles to look back for divergence checks

    def __init__(self):
        # CVD accumulator — resets daily
        self.cvd:          float = 0.0
        self._session_day: int   = -1
        self._session_vol: float = 0.0   # session total volume (for CVD% denominator)

        # Rolling history for divergence analysis
        self._cvd_history:   Deque[float] = deque(maxlen=self.LOOKBACK)
        self._close_history: Deque[float] = deque(maxlen=self.LOOKBACK)
        self._delta_history: Deque[float] = deque(maxlen=self.LOOKBACK)
        self._vol_history:   Deque[float] = deque(maxlen=self.LOOKBACK)

        # Per-candle CVD open/close (for CVD candles display)
        self._cvd_open: float = 0.0
        self.cvd_candle_open:  float = 0.0
        self.cvd_candle_close: float = 0.0

        # Signal flags (set after each update)
        self.absorption_detected:  bool = False
        self.absorption_direction: str  = ""   # 'LONG' or 'SHORT'
        self.exhaustion_detected:  bool = False
        self.exhaustion_direction: str  = ""

        # Recency counters — candles elapsed since each signal last fired.
        # Used by state_builder to produce a decaying recency feature instead
        # of a sparse binary.  Capped at 999 (= "not seen recently").
        self._candles_since_absorption: int = 999
        self._candles_since_exhaustion: int = 999

        # Latest delta values
        self.last_delta: float = 0.0
        self.avg_delta:  float = 0.0

    # ── Update ──────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        self._reset_session_if_needed(candle)

        self._cvd_open         = self.cvd
        self.cvd              += candle.delta
        self.cvd_candle_open   = self._cvd_open
        self.cvd_candle_close  = self.cvd
        self.last_delta        = candle.delta
        self._session_vol     += candle.volume

        self._cvd_history.append(self.cvd)
        self._close_history.append(candle.close)
        self._delta_history.append(candle.delta)
        self._vol_history.append(candle.volume)

        if self._vol_history:
            self.avg_delta = float(np.mean([abs(d) for d in self._delta_history]))

        self._detect_absorption(candle)
        self._detect_exhaustion()

        # Update recency counters (reset to 0 when signal fires, else increment)
        if self.absorption_detected:
            self._candles_since_absorption = 0
        else:
            self._candles_since_absorption = min(self._candles_since_absorption + 1, 999)

        if self.exhaustion_detected:
            self._candles_since_exhaustion = 0
        else:
            self._candles_since_exhaustion = min(self._candles_since_exhaustion + 1, 999)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def cvd_trend(self) -> str:
        """Short-term CVD momentum direction."""
        if len(self._cvd_history) < 3:
            return "NEUTRAL"
        hist = list(self._cvd_history)
        if hist[-1] > hist[-3]:
            return "BULLISH"
        if hist[-1] < hist[-3]:
            return "BEARISH"
        return "NEUTRAL"

    @property
    def cumulative_delta_pct(self) -> float:
        """Session CVD as a percentage of total session volume.
        Positive = net buying pressure, negative = net selling pressure."""
        if self._session_vol == 0:
            return 0.0
        return self.cvd / self._session_vol * 100.0

    # ── Signal Detection ────────────────────────────────────────────────────

    def _detect_absorption(self, candle: Candle) -> None:
        """
        Absorption: high volume but tiny price move AND tiny CVD change.

        High-volume buy (delta strongly positive) but price doesn't rise
          → sellers absorbing buyers → bearish (SHORT signal)

        High-volume sell (delta strongly negative) but price doesn't fall
          → buyers absorbing sellers → bullish (LONG signal)
        """
        if len(self._vol_history) < 5 or self.avg_delta == 0:
            self.absorption_detected  = False
            self.absorption_direction = ""
            return

        avg_vol = float(np.mean(list(self._vol_history)[:-1]))
        cvd_change = abs(candle.delta)  # CVD change this candle

        is_high_volume = candle.volume > avg_vol * 1.5
        is_small_body  = candle.body_size < _get_atr_proxy(list(self._close_history)) * config.ABSORPTION_BODY

        if not (is_high_volume and is_small_body):
            self.absorption_detected  = False
            self.absorption_direction = ""
            return

        # Buy absorption: large buy delta but no up move
        if candle.delta > self.avg_delta * 1.5 and candle.is_bullish is False:
            self.absorption_detected  = True
            self.absorption_direction = "SHORT"
        # Sell absorption: large sell delta but no down move
        elif candle.delta < -self.avg_delta * 1.5 and candle.is_bullish is True:
            self.absorption_detected  = True
            self.absorption_direction = "LONG"
        else:
            self.absorption_detected  = False
            self.absorption_direction = ""

    def _detect_exhaustion(self) -> None:
        """
        Exhaustion / Delta Divergence:
          Price makes new high but CVD is lower than previous high → SHORT
          Price makes new low but CVD is higher than previous low  → LONG
        """
        if len(self._cvd_history) < self.LOOKBACK:
            self.exhaustion_detected  = False
            self.exhaustion_direction = ""
            return

        closes = list(self._close_history)
        cvds   = list(self._cvd_history)

        # Find previous swing high
        prev_high_idx  = np.argmax(closes[:-2])
        prev_low_idx   = np.argmin(closes[:-2])
        current_close  = closes[-1]
        current_cvd    = cvds[-1]
        prev_high_cvd  = cvds[prev_high_idx]
        prev_low_cvd   = cvds[prev_low_idx]
        prev_high_px   = closes[prev_high_idx]
        prev_low_px    = closes[prev_low_idx]

        # Bearish exhaustion: new price high, lower CVD
        if (current_close > prev_high_px and current_cvd < prev_high_cvd * 0.95):
            self.exhaustion_detected  = True
            self.exhaustion_direction = "SHORT"
        # Bullish exhaustion: new price low, higher CVD
        elif (current_close < prev_low_px and current_cvd > prev_low_cvd * 0.95):
            self.exhaustion_detected  = True
            self.exhaustion_direction = "LONG"
        else:
            self.exhaustion_detected  = False
            self.exhaustion_direction = ""

    def _reset_session_if_needed(self, candle: Candle) -> None:
        from datetime import datetime, timezone
        day = datetime.fromtimestamp(candle.ts / 1000, tz=timezone.utc).day
        if day != self._session_day:
            self.cvd          = 0.0
            self._session_vol = 0.0
            self._session_day = day


def _get_atr_proxy(closes: list, period: int = 5) -> float:
    """Quick ATR proxy using close-to-close changes."""
    if len(closes) < 2:
        return 1.0
    changes = [abs(closes[i] - closes[i-1]) for i in range(1, min(len(closes), period+1))]
    return sum(changes) / len(changes) if changes else 1.0
