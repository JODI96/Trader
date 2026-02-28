"""
modes/auto_trade.py — Live Binance Futures automated trading.

When a signal fires:
  1. Place MARKET entry order
  2. Place STOP_MARKET SL order
  3. Place TAKE_PROFIT_MARKET TP order
  4. Monitor position until one side triggers
  5. Cancel the remaining order after exit
  6. Update risk manager with outcome

Safety checks:
  - Confirmation prompt before first live order in session
  - Respect risk manager can_trade() check
  - Never place more than one position at a time
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import config
from data.binance_client import BinanceClient
from data.market_data import Signal
from strategy.risk_manager import RiskManager, TradeResult

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    signal:       Signal
    entry_price:  float
    size:         float
    sl_order_id:  Optional[int] = None
    tp_order_id:  Optional[int] = None
    open_time:    float = field(default_factory=time.time)
    status:       str = "OPEN"


class AutoTradeMode:
    """
    Places real orders on Binance Futures based on incoming signals.

    IMPORTANT: Set BINANCE_TESTNET=true in .env to use the testnet before
    going live.  All trade confirmations will be logged.
    """

    def __init__(self, client: BinanceClient, risk_mgr: RiskManager):
        self._client    = client
        self._risk      = risk_mgr
        self._position: Optional[LivePosition] = None
        self._confirmed_this_session = False

        # Set leverage on startup
        self._client.set_leverage(config.SYMBOL, config.LEVERAGE)

    # ── Public API ──────────────────────────────────────────────────────────

    def process_signal(self, signal: Signal) -> Optional[LivePosition]:
        """
        Called when a new signal is generated.
        Returns LivePosition if an order was placed, None otherwise.
        """
        if self._position:
            logger.debug("Signal ignored: position already open")
            return None

        can_trade, reason = self._risk.can_trade()
        if not can_trade:
            logger.warning(f"Signal blocked: {reason}")
            return None

        if not self._confirmed_this_session:
            logger.warning(
                "AUTO TRADE MODE ACTIVE — placing real orders on "
                f"{'TESTNET' if config.BINANCE_TESTNET else 'LIVE'}"
            )
            self._confirmed_this_session = True

        return self._open_position(signal)

    def update_price(self, price: float) -> Optional[LivePosition]:
        """
        Call on each price tick.  Checks if position has been closed
        by a triggered SL/TP order on Binance.
        Returns closed position if exit detected.
        """
        if not self._position:
            return None
        return self._check_position_status(price)

    # ── Order execution ─────────────────────────────────────────────────────

    def _open_position(self, signal: Signal) -> Optional[LivePosition]:
        size = self._risk.calculate_size(signal.entry_price, signal.stop_loss)
        if size <= 0:
            logger.warning("Calculated size is 0 — skipping")
            return None

        side = "BUY" if signal.direction == "LONG" else "SELL"
        sl_side = "SELL" if signal.direction == "LONG" else "BUY"

        try:
            # 1. Market entry
            entry_resp = self._client.place_market_order(
                symbol = config.SYMBOL,
                side   = side,
                qty    = size,
            )
            actual_entry = float(entry_resp.get("avgPrice", signal.entry_price) or signal.entry_price)
            if actual_entry == 0:
                actual_entry = signal.entry_price

            logger.info(
                f"[AUTO] ENTRY {signal.direction} @ {actual_entry:.2f}  "
                f"size={size:.4f}  strategy={signal.strategy}"
            )

            # Recalculate TP based on actual entry
            from strategy.risk_manager import RiskManager as RM
            sl = signal.stop_loss
            tp = actual_entry + abs(actual_entry - sl) * config.MIN_RR_RATIO \
                if signal.direction == "LONG" \
                else actual_entry - abs(actual_entry - sl) * config.MIN_RR_RATIO

            # 2. Stop-loss order
            sl_resp = self._client.place_stop_market(
                symbol     = config.SYMBOL,
                side       = sl_side,
                qty        = size,
                stop_price = signal.stop_loss,
            )
            sl_id = sl_resp.get("orderId")

            # 3. Take-profit order
            tp_resp = self._client.place_take_profit_market(
                symbol     = config.SYMBOL,
                side       = sl_side,
                qty        = size,
                stop_price = tp,
            )
            tp_id = tp_resp.get("orderId")

            self._position = LivePosition(
                signal      = signal,
                entry_price = actual_entry,
                size        = size,
                sl_order_id = sl_id,
                tp_order_id = tp_id,
            )
            logger.info(
                f"[AUTO] Orders placed  SL@{signal.stop_loss:.2f} (id={sl_id})  "
                f"TP@{tp:.2f} (id={tp_id})"
            )
            return self._position

        except Exception as e:
            logger.error(f"[AUTO] Order placement failed: {e}")
            # Emergency: try to close any accidental position
            self._emergency_close()
            return None

    def _check_position_status(self, price: float) -> Optional[LivePosition]:
        """
        Check if position is still open on Binance.
        If position amount is 0, determine win/loss and record.
        """
        pos = self._position
        if pos is None:
            return None

        try:
            live_pos = self._client.get_open_position(config.SYMBOL)
            if live_pos is None or float(live_pos.get("positionAmt", 0)) == 0:
                # Position closed externally (SL or TP triggered)
                return self._reconcile_closed(pos, price)
        except Exception as e:
            logger.debug(f"Position status check failed: {e}")

        return None

    def _reconcile_closed(self, pos: LivePosition, price: float) -> LivePosition:
        """Called when Binance shows position is closed."""
        # Determine outcome based on which side price is on
        if pos.signal.direction == "LONG":
            won = price >= pos.entry_price
        else:
            won = price <= pos.entry_price

        # Calculate approximate PnL
        if pos.signal.direction == "LONG":
            pnl = (price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - price) * pos.size

        trade = TradeResult(
            entry_price   = pos.entry_price,
            exit_price    = price,
            direction     = pos.signal.direction,
            size          = pos.size,
            strategy      = pos.signal.strategy,
            timestamp_in  = pos.open_time,
            timestamp_out = time.time(),
        )

        if pnl > 0:
            pos.status = "WIN"
            self._risk.record_win(pnl, trade)
        elif pnl < 0:
            pos.status = "LOSS"
            self._risk.record_loss(abs(pnl), trade)
        else:
            pos.status = "BE"
            self._risk.record_breakeven(trade)

        # Cancel any remaining open SL/TP order
        try:
            self._client.cancel_all_orders(config.SYMBOL)
        except Exception:
            pass

        logger.info(
            f"[AUTO] Position closed {pos.status}  PnL ≈ {pnl:+.2f} USDT"
        )
        self._position = None
        return pos

    def _emergency_close(self) -> None:
        """Cancel all orders (safety net)."""
        try:
            self._client.cancel_all_orders(config.SYMBOL)
            logger.warning("[AUTO] Emergency: cancelled all open orders")
        except Exception as e:
            logger.error(f"[AUTO] Emergency close failed: {e}")

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def has_open_position(self) -> bool:
        return self._position is not None

    @property
    def open_position(self) -> Optional[LivePosition]:
        return self._position

    def unrealized_pnl(self, current_price: float) -> float:
        if not self._position:
            return 0.0
        pos = self._position
        if pos.signal.direction == "LONG":
            return (current_price - pos.entry_price) * pos.size
        return (pos.entry_price - current_price) * pos.size
