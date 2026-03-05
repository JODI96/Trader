"""
fetch_altcoins.py — Batch-Download 1m Klines für mehrere Coins.

Lädt 1m Kerzen von Binance Futures für die angegebenen Coins und Jahre,
und speichert sie im selben Format wie die BTC-Daten:

  data/historical/<SYMBOL>/
    full_year/   — YYYY_1m.csv
    monthly/     — YYYY-MM_1m.csv
    seasonal/    — season_YYYY_1m.csv

Usage:
    python fetch_altcoins.py
"""
from __future__ import annotations

import calendar
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Projektpfad hinzufügen damit Imports funktionieren
sys.path.insert(0, str(Path(__file__).parent))
from data.historical_store import DatasetManager

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Konfiguration ─────────────────────────────────────────────────────────────

COINS = [
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "HBARUSDT",
    "DOTUSDT",
    "XRPUSDT",
]

YEARS = [2023, 2024, 2025]

_KLINES_URL  = "https://fapi.binance.com/fapi/v1/klines"
_KLINE_LIMIT = 1500


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_1m(symbol: str, start_ms: int, end_ms: int) -> list:
    """Lädt alle 1m Kerzen für den angegebenen Zeitraum."""
    all_klines: list = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  "1m",
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
                print(f"      Fehler (Versuch {attempt + 1}/3): {e}")
                time.sleep(5)
        else:
            print("      Abbruch nach 3 Fehlversuchen.")
            break

        if not batch:
            break

        all_klines.extend(batch)
        current = int(batch[-1][6]) + 1  # next start = last close_time + 1 ms

        if len(batch) < _KLINE_LIMIT:
            break

        time.sleep(0.1)  # API schonen

    return all_klines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now = datetime.now(timezone.utc)
    total = len(COINS) * len(YEARS)
    done  = 0

    print(f"\n{'-' * 60}")
    print(f"  Altcoin 1m Batch-Download")
    print(f"  Coins : {', '.join(c.replace('USDT','') for c in COINS)}")
    print(f"  Jahre : {', '.join(str(y) for y in YEARS)}")
    print(f"  Gesamt: {total} Kombinationen")
    print(f"{'-' * 60}\n")

    for symbol in COINS:
        mgr = DatasetManager(symbol)

        for year in YEARS:
            done += 1
            start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
            end_dt   = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            if end_dt > now:
                end_dt = now

            start_ms = int(start_dt.timestamp() * 1000)
            end_ms   = int(end_dt.timestamp() * 1000)

            label = f"{symbol} {year}"
            print(f"[{done}/{total}]  {label} …", flush=True)
            print(f"        Lade 1m Kerzen ({start_dt.date()} – {end_dt.date()}) …", end=" ", flush=True)

            klines = fetch_1m(symbol, start_ms, end_ms)

            if not klines:
                print("KEINE DATEN — übersprungen")
                continue

            print(f"{len(klines):,} Kerzen", flush=True)
            print(f"        Speichere …", end=" ", flush=True)

            # Nur 1m speichern, 5m/15m leer lassen
            counts = mgr.save_all(klines, [], [])

            print(
                f"OK  "
                f"({counts['full_year']} full_year, "
                f"{counts['monthly']} monthly, "
                f"{counts['seasonal']} seasonal)"
            )

        print()

    print(f"{'-' * 60}")
    print(f"  Fertig! Alle Daten unter: data/historical/")
    print(f"{'-' * 60}\n")


if __name__ == "__main__":
    main()
