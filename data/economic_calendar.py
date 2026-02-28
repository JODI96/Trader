"""
economic_calendar.py — Fetches upcoming economic events.

Primary source: ForexFactory public JSON API (no auth, no blocks)
  https://nfs.faireconomy.media/ff_calendar_thisweek.json

Fallback: basic hardcoded awareness that the filter should stay permissive
when data is unavailable.

Results cached for 60 minutes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import requests

import config

logger = logging.getLogger(__name__)

FF_URL      = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_NEXT_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Retry on failure: wait this many seconds before trying again
RETRY_DELAY = 300   # 5 minutes


@dataclass
class EconomicEvent:
    dt: datetime          # UTC datetime of the event
    name: str
    country: str
    impact: str           # 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN'
    actual: str = ""
    forecast: str = ""

    @property
    def is_high_impact(self) -> bool:
        return self.impact == "HIGH"

    @property
    def minutes_until(self) -> float:
        now = datetime.now(tz=timezone.utc)
        return (self.dt - now).total_seconds() / 60.0


class EconomicCalendar:
    """
    Fetches and caches economic events from ForexFactory.
    Fails gracefully — missing data never blocks trading.
    """

    def __init__(self):
        self._events:      List[EconomicEvent] = []
        self._last_fetch:  float = 0.0
        self._last_error:  float = 0.0   # timestamp of last failed fetch
        self._cache_ttl    = 3600        # 1 hour cache
        self._fetch_count  = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def get_events(self, refresh: bool = False) -> List[EconomicEvent]:
        now = time.time()
        cache_stale   = (now - self._last_fetch) > self._cache_ttl
        retry_ok      = (now - self._last_error) > RETRY_DELAY

        if refresh or (cache_stale and retry_ok):
            self._fetch()
        return self._events

    def is_news_blackout(self) -> bool:
        """
        Return True if a HIGH-impact event is within ±NEWS_BUFFER_MIN minutes.
        Returns False (trading allowed) when data is unavailable.
        """
        try:
            events = self.get_events()
        except Exception:
            return False
        buf = config.NEWS_BUFFER_MIN
        for ev in events:
            if ev.is_high_impact:
                mins = ev.minutes_until
                if -buf <= mins <= buf:
                    return True
        return False

    def next_high_impact(self) -> Optional[EconomicEvent]:
        """Return the next upcoming HIGH-impact event (or None)."""
        now = datetime.now(tz=timezone.utc)
        upcoming = [
            e for e in self._events
            if e.is_high_impact and e.dt > now
        ]
        return min(upcoming, key=lambda e: e.dt) if upcoming else None

    # ── Internal ────────────────────────────────────────────────────────────

    def _fetch(self) -> None:
        for url in [FF_URL, FF_NEXT_URL]:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                resp.raise_for_status()
                parsed = self._parse_ff(resp.json())
                if url == FF_URL:
                    self._events = parsed
                else:
                    # Append next-week events, avoid duplicates
                    existing_names = {e.name + str(e.dt) for e in self._events}
                    for ev in parsed:
                        if ev.name + str(ev.dt) not in existing_names:
                            self._events.append(ev)
                self._fetch_count += 1
                logger.info(
                    f"Economic calendar: {len(self._events)} events loaded "
                    f"(fetch #{self._fetch_count})"
                )
            except Exception as e:
                logger.warning(f"Economic calendar fetch failed ({url}): {e}")
                self._last_error = time.time()

        # Always update last_fetch to prevent retry spam
        self._last_fetch = time.time()

    def _parse_ff(self, data: list) -> List[EconomicEvent]:
        """
        Parse ForexFactory JSON.
        API returns ISO 8601 dates with timezone offset:
          {
            "title": "ISM Manufacturing PMI",
            "country": "USD",
            "date": "2026-03-03T10:00:00-05:00",
            "impact": "High",
            "forecast": "49.5",
            "previous": "50.9"
          }
        """
        events: List[EconomicEvent] = []
        for item in data:
            try:
                impact_raw = item.get("impact", "").strip().lower()
                impact = {
                    "high":    "HIGH",
                    "medium":  "MEDIUM",
                    "low":     "LOW",
                    "holiday": "LOW",
                }.get(impact_raw, "UNKNOWN")

                date_str = item.get("date", "")
                dt       = _parse_iso_datetime(date_str)

                events.append(EconomicEvent(
                    dt       = dt,
                    name     = item.get("title", ""),
                    country  = item.get("country", ""),
                    impact   = impact,
                    forecast = str(item.get("forecast", "")),
                    actual   = str(item.get("actual", "")),
                ))
            except Exception:
                continue
        return events


# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_iso_datetime(date_str: str) -> datetime:
    """
    Parse ISO 8601 datetime string with timezone offset to UTC datetime.
    e.g. "2026-03-03T10:00:00-05:00"
    """
    from datetime import timedelta

    # Python 3.7+ fromisoformat handles offset-aware strings
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.astimezone(timezone.utc)
    except Exception:
        # Fallback: strip offset and treat as UTC
        base = date_str[:19]
        return datetime.fromisoformat(base).replace(tzinfo=timezone.utc)
