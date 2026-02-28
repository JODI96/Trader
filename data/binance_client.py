"""
binance_client.py — Binance Futures REST + direct WebSocket.

The WebSocket uses websocket-client directly (not python-binance's
ThreadedWebsocketManager which has a 100-message internal queue limit that
overflows during active BTC trading).

Stream URL: wss://fstream.binance.com/ws/<symbol>@aggTrade
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable, Optional

import websocket
from binance.client import Client

import config
from data.market_data import RawTrade

logger = logging.getLogger(__name__)

# Binance Futures aggTrade stream endpoint (public, no auth needed)
WS_BASE = "wss://fstream.binance.com/ws/{symbol}@aggTrade"


class BinanceClient:
    """
    Wraps python-binance.  All REST calls go through `rest`, all WebSocket
    subscription management goes through this class.
    """

    def __init__(self):
        tld = "com"
        self.rest = Client(
            api_key    = config.BINANCE_API_KEY,
            api_secret = config.BINANCE_API_SECRET,
            testnet    = config.BINANCE_TESTNET,
            tld        = tld,
        )
        if config.BINANCE_TESTNET:
            self.rest.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            logger.info("Using Binance Futures TESTNET")
        else:
            logger.info("Using Binance Futures LIVE — real money!")

        self._trade_callbacks: list[Callable[[RawTrade], None]] = []
        self._ws_running = False

    # ── Callbacks ───────────────────────────────────────────────────────────

    def add_trade_callback(self, cb: Callable[[RawTrade], None]) -> None:
        self._trade_callbacks.append(cb)

    # ── REST helpers ────────────────────────────────────────────────────────

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> list:
        """Fetch klines (OHLCV) — includes taker buy volume in index 9."""
        return self.rest.futures_klines(symbol=symbol, interval=interval, limit=limit)

    def get_futures_balance(self, asset: str = "USDT") -> float:
        try:
            balances = self.rest.futures_account_balance()
            for b in balances:
                if b["asset"] == asset:
                    return float(b["availableBalance"])
        except Exception as e:
            logger.warning(f"Could not fetch futures balance: {e}")
        return 0.0

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.rest.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")

    def place_market_order(self, symbol: str, side: str, qty: float) -> dict:
        """side: 'BUY' or 'SELL'"""
        return self.rest.futures_create_order(
            symbol   = symbol,
            side     = side,
            type     = "MARKET",
            quantity = f"{qty:.3f}",
        )

    def place_stop_market(self, symbol: str, side: str, qty: float,
                          stop_price: float) -> dict:
        return self.rest.futures_create_order(
            symbol        = symbol,
            side          = side,
            type          = "STOP_MARKET",
            quantity      = f"{qty:.3f}",
            stopPrice     = f"{stop_price:.2f}",
            closePosition = "true",
        )

    def place_take_profit_market(self, symbol: str, side: str, qty: float,
                                  stop_price: float) -> dict:
        return self.rest.futures_create_order(
            symbol        = symbol,
            side          = side,
            type          = "TAKE_PROFIT_MARKET",
            quantity      = f"{qty:.3f}",
            stopPrice     = f"{stop_price:.2f}",
            closePosition = "true",
        )

    def cancel_all_orders(self, symbol: str) -> None:
        try:
            self.rest.futures_cancel_all_open_orders(symbol=symbol)
        except Exception as e:
            logger.warning(f"Could not cancel orders: {e}")

    def get_open_position(self, symbol: str) -> Optional[dict]:
        try:
            positions = self.rest.futures_position_information(symbol=symbol)
            for p in positions:
                if float(p["positionAmt"]) != 0:
                    return p
        except Exception as e:
            logger.warning(f"Could not fetch position: {e}")
        return None

    def get_ticker_price(self, symbol: str) -> float:
        try:
            t = self.rest.futures_symbol_ticker(symbol=symbol)
            return float(t["price"])
        except Exception:
            return 0.0

    # ── WebSocket ───────────────────────────────────────────────────────────

    def start_stream(self, symbol: str) -> None:
        self._ws_symbol  = symbol.lower()
        self._ws_running = True
        url = WS_BASE.format(symbol=self._ws_symbol)

        def on_message(ws, raw: str) -> None:
            try:
                data = json.loads(raw)
                if data.get("e") != "aggTrade":
                    return
                trade = RawTrade(
                    price        = float(data["p"]),
                    qty          = float(data["q"]),
                    is_buy       = not bool(data["m"]),
                    timestamp_ms = int(data["T"]),
                )
                for cb in self._trade_callbacks:
                    cb(trade)
            except Exception as e:
                logger.debug(f"aggTrade parse error: {e}")

        def on_error(ws, error) -> None:
            logger.warning(f"WebSocket error: {error}")

        def on_close(ws, code, msg) -> None:
            logger.info(f"WebSocket closed (code={code}) — reconnecting in 3s")
            if self._ws_running:
                time.sleep(3)
                self.start_stream(self._ws_symbol)

        def on_open(ws) -> None:
            logger.info(f"aggTrade stream connected: {url}")

        self._ws = websocket.WebSocketApp(
            url,
            on_message = on_message,
            on_error   = on_error,
            on_close   = on_close,
            on_open    = on_open,
        )
        self._ws_thread = threading.Thread(
            target = lambda: self._ws.run_forever(ping_interval=20, ping_timeout=10),
            daemon = True,
            name   = "ws-aggtrade",
        )
        self._ws_thread.start()
        logger.info(f"aggTrade WebSocket thread started for {symbol}")

    def stop_stream(self) -> None:
        self._ws_running = False
        if hasattr(self, "_ws"):
            try:
                self._ws.close()
            except Exception:
                pass
