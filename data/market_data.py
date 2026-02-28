"""
market_data.py — Core data structures and 15s / 15min candle aggregation.

CandleBuilder accepts raw RawTrade events (from aggTrade WebSocket stream)
and emits completed Candle objects when a time-bucket boundary is crossed.
MarketDataStore keeps rolling deques of completed candles for both timeframes.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional

import config


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RawTrade:
    price: float
    qty: float
    is_buy: bool          # True = aggressive buy  (isBuyerMaker=False in Binance msg)
    timestamp_ms: int


@dataclass
class Candle:
    ts: int               # bucket start timestamp (ms)
    open: float
    high: float
    low: float
    close: float
    volume: float         # total base-asset volume
    buy_vol: float        # aggressive buy volume
    sell_vol: float       # aggressive sell volume
    delta: float          # buy_vol - sell_vol  (positive = bullish pressure)
    num_trades: int

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def delta_pct(self) -> float:
        return self.delta / self.volume if self.volume > 0 else 0.0

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


@dataclass
class Signal:
    timestamp: float
    direction: str         # 'LONG' | 'SHORT'
    strategy: str          # human-readable strategy name
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    confidence: int        # 1–3  (number of confluences met)
    reason: str            # detailed explanation string


# ──────────────────────────────────────────────────────────────────────────────
# Candle builder
# ──────────────────────────────────────────────────────────────────────────────

class CandleBuilder:
    """
    Aggregates RawTrade events into fixed-interval candles.
    Calls `on_candle(candle)` every time a candle is *completed*
    (i.e. when the next trade falls into a new time bucket).
    """

    def __init__(self, interval_sec: int, on_candle: Callable[[Candle], None]):
        self.interval_ms  = interval_sec * 1000
        self.on_candle    = on_candle
        self._bucket_ts: Optional[int] = None
        self._open = self._high = self._low = self._close = 0.0
        self._volume = self._buy_vol = self._sell_vol = 0.0
        self._num_trades = 0

    def process(self, trade: RawTrade) -> None:
        bucket = (trade.timestamp_ms // self.interval_ms) * self.interval_ms

        if self._bucket_ts is None:
            # First trade ever
            self._start_candle(bucket, trade)
            return

        if bucket != self._bucket_ts:
            # Emit completed candle
            self._emit()
            self._start_candle(bucket, trade)
        else:
            # Update current candle
            self._high = max(self._high, trade.price)
            self._low  = min(self._low,  trade.price)
            self._close = trade.price
            self._volume += trade.qty
            if trade.is_buy:
                self._buy_vol += trade.qty
            else:
                self._sell_vol += trade.qty
            self._num_trades += 1

    def _start_candle(self, bucket: int, trade: RawTrade) -> None:
        self._bucket_ts  = bucket
        self._open       = trade.price
        self._high       = trade.price
        self._low        = trade.price
        self._close      = trade.price
        self._volume     = trade.qty
        self._buy_vol    = trade.qty if trade.is_buy else 0.0
        self._sell_vol   = 0.0 if trade.is_buy else trade.qty
        self._num_trades = 1

    def _emit(self) -> None:
        candle = Candle(
            ts         = self._bucket_ts,
            open       = self._open,
            high       = self._high,
            low        = self._low,
            close      = self._close,
            volume     = self._volume,
            buy_vol    = self._buy_vol,
            sell_vol   = self._sell_vol,
            delta      = self._buy_vol - self._sell_vol,
            num_trades = self._num_trades,
        )
        self.on_candle(candle)

    def flush(self) -> None:
        """Emit whatever partial candle is in progress (call on shutdown)."""
        if self._bucket_ts is not None and self._num_trades > 0:
            self._emit()
            self._bucket_ts = None


# ──────────────────────────────────────────────────────────────────────────────
# Market data store
# ──────────────────────────────────────────────────────────────────────────────

class MarketDataStore:
    """
    Rolling buffers of completed candles for the scalp (15s) and HTF (15min)
    timeframes.  Subscribers register callbacks that fire on each new candle.
    """

    def __init__(self):
        self.scalp_candles: Deque[Candle] = deque(maxlen=config.LOOKBACK_CANDLES)
        self.htf_candles:   Deque[Candle] = deque(maxlen=config.HTF_LOOKBACK)

        self._scalp_callbacks: List[Callable[[Candle], None]] = []
        self._htf_callbacks:   List[Callable[[Candle], None]] = []

        # Builders for both timeframes
        self._scalp_builder = CandleBuilder(config.SCALP_TF_SEC,  self._on_scalp_candle)
        self._htf_builder   = CandleBuilder(config.HTF_TF_SEC,    self._on_htf_candle)

        # Latest price (updated on every trade, not just candle close)
        self.last_price: float = 0.0
        self.last_trade_ts: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def on_scalp(self, cb: Callable[[Candle], None]) -> None:
        self._scalp_callbacks.append(cb)

    def on_htf(self, cb: Callable[[Candle], None]) -> None:
        self._htf_callbacks.append(cb)

    def feed_trade(self, trade: RawTrade) -> None:
        self.last_price    = trade.price
        self.last_trade_ts = trade.timestamp_ms / 1000.0
        self._scalp_builder.process(trade)
        self._htf_builder.process(trade)

    def seed_from_klines(self, klines: list, timeframe: str = "scalp") -> None:
        """
        Pre-fill history from Binance REST kline data so indicators have
        enough warmup data before the live stream starts.

        klines: raw list from client.futures_klines()
        Each entry: [open_time, open, high, low, close, volume, ...,
                     taker_buy_base_asset_volume, ...]
        """
        target = self.scalp_candles if timeframe == "scalp" else self.htf_candles
        for k in klines:
            vol      = float(k[5])
            buy_vol  = float(k[9])     # taker buy base asset volume
            sell_vol = vol - buy_vol
            candle   = Candle(
                ts         = int(k[0]),
                open       = float(k[1]),
                high       = float(k[2]),
                low        = float(k[3]),
                close      = float(k[4]),
                volume     = vol,
                buy_vol    = buy_vol,
                sell_vol   = sell_vol,
                delta      = buy_vol - sell_vol,
                num_trades = int(k[8]),
            )
            target.append(candle)
        if target:
            self.last_price = list(target)[-1].close

    def flush(self) -> None:
        self._scalp_builder.flush()
        self._htf_builder.flush()

    # ── Internal ────────────────────────────────────────────────────────────

    def _on_scalp_candle(self, candle: Candle) -> None:
        self.scalp_candles.append(candle)
        for cb in self._scalp_callbacks:
            try:
                cb(candle)
            except Exception as e:
                import logging, traceback
                logging.getLogger(__name__).error(
                    f"Scalp candle callback error: {e}\n{traceback.format_exc()}"
                )

    def _on_htf_candle(self, candle: Candle) -> None:
        self.htf_candles.append(candle)
        for cb in self._htf_callbacks:
            try:
                cb(candle)
            except Exception as e:
                import logging, traceback
                logging.getLogger(__name__).error(
                    f"HTF candle callback error: {e}\n{traceback.format_exc()}"
                )
