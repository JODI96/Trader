"""
filters.py — Pre-trade filters that gate signal execution.

All filters must pass for a trade to be allowed:
  1. Volume filter   — current volume must be ≥ 50% of rolling average
  2. News filter     — no HIGH-impact event within ±15 minutes (skipped in backtest)
  3. Session filter  — avoid 12:00–14:00 UTC low-liquidity hours
  4. Volatility      — ATR must be above minimum threshold
  5. Session losses  — risk manager must allow trading
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

import config
from data.economic_calendar import EconomicCalendar
from indicators.atr import ATRIndicator
from indicators.delta_volume import DeltaVolumeIndicator
from strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class TradeFilters:
    """
    Aggregates all pre-trade checks.  Call `check_all()` before every potential
    trade entry.  Returns (allowed: bool, reason: str).
    """

    def __init__(
        self,
        atr: ATRIndicator,
        delta_vol: DeltaVolumeIndicator,
        risk_mgr: RiskManager,
        calendar: EconomicCalendar,
        backtest_mode: bool = False,
    ):
        self._atr          = atr
        self._dv           = delta_vol
        self._risk         = risk_mgr
        self._calendar     = calendar
        self._backtest_mode = backtest_mode

    # ── Master check ────────────────────────────────────────────────────────

    def check_all(self, candle_ts_ms: int = 0) -> Tuple[bool, str]:
        """
        Run all filters in order.  Returns (True, "") if trading is OK,
        or (False, reason_string) on the first failed filter.

        candle_ts_ms: candle open timestamp in milliseconds UTC.  Pass
        candle.ts so the session-time filter uses the historical candle time
        rather than wall-clock time (critical for correct backtest behaviour).
        In live mode candle.ts ≈ now, so the result is identical either way.
        """
        ok, reason = self._check_session_losses()
        if not ok:
            return False, reason

        # News filter is irrelevant for historical replays — today's economic
        # events have no bearing on price action from weeks or months ago.
        if not self._backtest_mode:
            ok, reason = self._check_news()
            if not ok:
                return False, reason

        ok, reason = self._check_session_time(candle_ts_ms)
        if not ok:
            return False, reason

        ok, reason = self._check_volume()
        if not ok:
            return False, reason

        ok, reason = self._check_volatility()
        if not ok:
            return False, reason

        return True, ""

    # ── Individual filters ──────────────────────────────────────────────────

    def _check_session_losses(self) -> Tuple[bool, str]:
        return self._risk.can_trade()

    def _check_news(self) -> Tuple[bool, str]:
        try:
            if self._calendar.is_news_blackout():
                ev = self._calendar.next_high_impact()
                name = ev.name if ev else "HIGH-impact event"
                return False, f"News blackout: {name}"
        except Exception as e:
            logger.debug(f"News filter skipped (error): {e}")
        return True, ""

    def _check_session_time(self, candle_ts_ms: int = 0) -> Tuple[bool, str]:
        """
        Avoid the low-liquidity 12:00–14:00 UTC window.
        Uses the candle timestamp when provided (backtest), else wall clock (live).
        """
        if candle_ts_ms > 0:
            dt_utc = datetime.fromtimestamp(candle_ts_ms / 1000.0, tz=timezone.utc)
        else:
            dt_utc = datetime.now(tz=timezone.utc)
        h = dt_utc.hour
        lo, hi = config.LOW_LIQUIDITY_UTC
        if lo <= h < hi:
            return False, f"Low-liquidity session ({h:02d}:xx UTC)"
        return True, ""

    def _check_volume(self) -> Tuple[bool, str]:
        avg = self._dv.avg_volume
        cur = self._dv.last_volume
        if avg > 0 and cur < avg * config.LOW_VOLUME_FACTOR:
            return False, f"Low volume ({cur:.2f} < {avg * config.LOW_VOLUME_FACTOR:.2f})"
        return True, ""

    def _check_volatility(self) -> Tuple[bool, str]:
        if not self._atr.is_ready:
            return False, "ATR not ready (warming up)"
        if self._atr.volatility_state == "LOW":
            return False, f"Low volatility (ATR {self._atr.atr_pct:.3f}%)"
        return True, ""

    # ── Individual filter results (for dashboard display) ───────────────────

    def status_dict(self) -> dict:
        return {
            "session_ok":  self._risk.can_trade()[0],
            "news_ok":     not self._calendar.is_news_blackout(),
            "session_time_ok": self._is_good_session_time(),
            "volume_ok":   self._is_volume_ok(),
            "volatility_ok": self._atr.volatility_state != "LOW",
        }

    def _is_good_session_time(self) -> bool:
        h = datetime.now(tz=timezone.utc).hour
        lo, hi = config.LOW_LIQUIDITY_UTC
        return not (lo <= h < hi)

    def _is_volume_ok(self) -> bool:
        avg = self._dv.avg_volume
        cur = self._dv.last_volume
        return avg == 0 or cur >= avg * config.LOW_VOLUME_FACTOR
