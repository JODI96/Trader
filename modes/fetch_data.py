"""
modes/fetch_data.py — Historical data downloader.

Downloads 1m, 5m, and 15m Binance Futures klines for a specific year/month
(or a rolling window of days) and saves them locally for offline backtesting.

Saved structure:
  data/historical/<SYMBOL>/
    full_year/   — YYYY_<tf>.csv  (use with --year 2025)
    monthly/     — YYYY-MM_<tf>.csv  (use with --year 2025 --month 1)
    seasonal/    — season_YYYY_<tf>.csv  (use with --data winter)

Usage:
    python main.py --mode fetch                        # interactive prompt
    python main.py --mode fetch --year 2025            # full year 2025
    python main.py --mode fetch --year 2025 --month 1  # January 2025
    python main.py --mode fetch --days 90              # last 90 days
    python main.py --mode fetch --symbol ETHUSDT --year 2024
"""
from __future__ import annotations

import calendar
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

import config
from data.historical_store import DatasetManager, SEASON_LABEL

logger = logging.getLogger(__name__)

_KLINES_URL  = "https://fapi.binance.com/fapi/v1/klines"
_KLINE_LIMIT = 1500   # Binance max per request


class DataFetcher:
    """Downloads and organises historical klines for offline backtesting."""

    def __init__(self, symbol: Optional[str] = None) -> None:
        self.symbol = (symbol or config.SYMBOL).upper()
        self.mgr    = DatasetManager(self.symbol)

    # ── Public entrypoint ────────────────────────────────────────────────────

    def run(
        self,
        days:  Optional[int] = None,
        year:  Optional[int] = None,
        month: Optional[int] = None,
    ) -> None:
        try:
            from rich.console import Console
            from rich.prompt import IntPrompt, Prompt
            con = Console()
        except ImportError:
            self._run_plain(days, year, month)
            return

        con.print(f"\n[bold bright_blue]Data Fetcher — {self.symbol}[/bold bright_blue]\n")
        con.print("  Downloads Binance Futures klines (1m / 5m / 15m) and saves them")
        con.print("  locally so backtest runs without hitting the API.\n")

        # Show existing datasets
        avail = self.mgr.list_datasets()
        if any(avail.values()):
            con.print("[yellow]Existing local datasets:[/yellow]")
            if avail["full_year"]:
                con.print(f"  Years    : {', '.join(avail['full_year'])}")
            if avail["monthly"]:
                con.print(f"  Monthly  : {', '.join(avail['monthly'])}")
            if avail["seasonal"]:
                labels = [SEASON_LABEL.get(s, s) for s in avail["seasonal"]]
                con.print(f"  Seasonal : {', '.join(labels)}")
            con.print()

        # If year/month/days were passed via CLI, skip the prompt
        if year is None and days is None:
            con.print("  [cyan]1[/cyan]  Fetch a specific year  (e.g. 2025)")
            con.print("  [cyan]2[/cyan]  Fetch a specific month (e.g. January 2025)")
            con.print("  [cyan]3[/cyan]  Fetch last N days\n")
            choice = Prompt.ask("Select", choices=["1", "2", "3"], default="1")

            if choice == "1":
                year = IntPrompt.ask("Year", default=datetime.now(timezone.utc).year)
            elif choice == "2":
                year  = IntPrompt.ask("Year",  default=datetime.now(timezone.utc).year)
                month = IntPrompt.ask("Month (1–12)", default=datetime.now(timezone.utc).month)
                month = max(1, min(12, month))
            else:
                days = IntPrompt.ask("How many days? (1–365)", default=90)
                days = max(1, min(365, days))

        start_ms, end_ms, label = self._resolve_range(days, year, month)
        con.print(f"\n  Fetching [bold]{label}[/bold] for [bold]{self.symbol}[/bold]…\n")
        self._fetch_and_save(start_ms, end_ms, label, con)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_plain(
        self,
        days:  Optional[int],
        year:  Optional[int],
        month: Optional[int],
    ) -> None:
        start_ms, end_ms, label = self._resolve_range(days, year, month)
        print(f"Fetching {label} for {self.symbol}…")
        self._fetch_and_save(start_ms, end_ms, label)

    def _resolve_range(
        self,
        days:  Optional[int],
        year:  Optional[int],
        month: Optional[int],
    ) -> tuple[int, int, str]:
        """
        Return (start_ms, end_ms, human_label) from whichever args were given.
        Priority: year/month > days > default 365 days.
        """
        now = datetime.now(timezone.utc)

        if year and month:
            month = max(1, min(12, month))
            last_day = calendar.monthrange(year, month)[1]
            start = datetime(year, month, 1,  tzinfo=timezone.utc)
            end   = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
            # Don't fetch into the future
            if end > now:
                end = now
            label = f"{start.strftime('%B %Y')}"
            return int(start.timestamp() * 1000), int(end.timestamp() * 1000), label

        if year:
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end   = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            if end > now:
                end = now
            label = f"Full year {year}"
            return int(start.timestamp() * 1000), int(end.timestamp() * 1000), label

        # Rolling days window
        n = days or 365
        end_ms   = int(now.timestamp() * 1000)
        start_ms = end_ms - n * 24 * 3600 * 1000
        return start_ms, end_ms, f"last {n} days"

    def _fetch_and_save(
        self,
        start_ms: int,
        end_ms:   int,
        label:    str,
        con=None,
    ) -> None:
        def _print(msg: str, end: str = "\n") -> None:
            if con:
                con.print(msg, end=end)
            else:
                print(msg, end=end, flush=True)

        klines_1m  = self._fetch("1m",  start_ms, end_ms, _print)
        klines_5m  = self._fetch("5m",  start_ms, end_ms, _print)
        klines_15m = self._fetch("15m", start_ms, end_ms, _print)

        if not klines_1m:
            _print("\n  [ERROR] No 1m data fetched. Check your internet connection.")
            return

        _print("\n  Saving to local storage…")
        counts = self.mgr.save_all(klines_1m, klines_5m, klines_15m)

        _print(f"\n  Done! ({label})")
        _print(f"  Location : data/historical/{self.symbol}/")
        _print(f"  Files    : {counts['full_year']} full-year, "
               f"{counts['monthly']} monthly, {counts['seasonal']} seasonal\n")

        avail = self.mgr.list_datasets()
        _print("  Use in backtest:")
        if avail["full_year"]:
            yr = avail["full_year"][-1]
            _print(f"    python main.py --mode backtest --year {yr}")
        if avail["monthly"]:
            ym = avail["monthly"][-1]
            yr, mo = ym.split("-")
            _print(f"    python main.py --mode backtest --year {yr} --month {int(mo)}")
        if avail["seasonal"]:
            _print(f"    python main.py --mode backtest --data {avail['seasonal'][0]}")
        _print("")

    def _fetch(self, interval: str, start_ms: int, end_ms: int, _print) -> list:
        _print(f"  Fetching {interval} klines…", end=" ")
        all_klines: list = []
        current = start_ms

        while current < end_ms:
            params = {
                "symbol":    self.symbol,
                "interval":  interval,
                "limit":     _KLINE_LIMIT,
                "startTime": current,
                "endTime":   end_ms,
            }
            for attempt in range(3):
                try:
                    resp = requests.get(_KLINES_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    batch = resp.json()
                    break
                except Exception as e:
                    logger.warning(f"fetch error (attempt {attempt + 1}/3): {e}")
                    time.sleep(5)
            else:
                logger.error("Failed to fetch after 3 attempts — stopping pagination")
                break

            if not batch:
                break

            all_klines.extend(batch)
            current = int(batch[-1][6]) + 1   # next start = last close_time + 1 ms

            if len(batch) < _KLINE_LIMIT:
                break

            time.sleep(0.1)   # be polite to the API

        _print(f"{len(all_klines):,} candles")
        return all_klines
