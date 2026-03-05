"""
modes/simulation.py — Paper trading simulation mode.

Manages a virtual account:
  - Accepts signals and "opens" virtual positions
  - Marks positions as WIN/LOSS/BE when price hits TP/SL
  - Tracks full PnL history, win rate, max drawdown
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import config
from data.market_data import Signal
from strategy.risk_manager import RiskManager, TradeResult

STATE_FILE = Path("sim_state.json")

logger = logging.getLogger(__name__)


@dataclass
class SimPosition:
    id:          str
    direction:   str        # 'LONG' | 'SHORT'
    entry_price: float
    stop_loss:   float
    take_profit: float
    size:        float      # base units (BTC)
    strategy:    str
    open_time:   float
    leverage:    float      # effective leverage when position was opened
    status:      str = "OPEN"   # OPEN | WIN | LOSS | BE
    close_price: float = 0.0
    close_time:  float = 0.0
    pnl_usdt:    float = 0.0   # net PnL after fees
    fee_usdt:    float = 0.0   # total round-trip fee paid
    # Break-even stop: once price reaches be_trigger, SL slides to be_sl
    # be_trigger = entry ± 1×SL_dist  (the 1:1 point)
    # be_sl      = entry ± fee/unit   (exactly covers round-trip cost)
    be_trigger:    float = 0.0
    be_sl:         float = 0.0
    breakeven_set: bool  = False

    @property
    def unrealized_pnl(self) -> float:
        """Calculated outside once we have the current price."""
        return 0.0


class SimulationMode:
    """
    Paper-trading engine.  Feed it signals and price ticks;
    it manages virtual positions and notifies on close.
    """

    def __init__(self, risk_mgr: RiskManager):
        self._risk = risk_mgr
        self.positions:    List[SimPosition] = []   # open
        self.closed:       List[SimPosition] = []   # history
        self._pos_counter: int = 0

        self.current_price:  float = 0.0
        self.peak_balance:   float = risk_mgr.balance
        self.max_drawdown:   float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def process_signal(self, signal: Signal) -> Optional[SimPosition]:
        """
        Open a virtual position from a signal.
        Returns the SimPosition, or None if rejected (e.g. position already open).
        """
        # Only one position at a time
        if self.positions:
            logger.debug("Signal ignored: position already open")
            return None

        can_trade, reason = self._risk.can_trade()
        if not can_trade:
            logger.warning(f"Signal rejected by risk manager: {reason}")
            return None

        size = self._risk.calculate_size(signal.entry_price, signal.stop_loss)
        if size <= 0:
            return None

        self._pos_counter += 1
        pos = SimPosition(
            id          = f"SIM-{self._pos_counter:04d}",
            direction   = signal.direction,
            entry_price = signal.entry_price,
            stop_loss   = signal.stop_loss,
            take_profit = signal.take_profit,
            size        = size,
            strategy    = signal.strategy,
            open_time   = signal.timestamp,
            leverage    = config.LEVERAGE,
        )

        # Break-even stop levels:
        #   be_trigger = entry ± 1×SL_dist  (when the trade is 1R in profit)
        #   be_sl      = entry ± fee/unit    (close here → PnL exactly covers fees)
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        fee_per_unit = signal.entry_price * config.FEE_TAKER / config.LEVERAGE
        if signal.direction == "LONG":
            pos.be_trigger = signal.entry_price + sl_dist
            pos.be_sl      = signal.entry_price + fee_per_unit
        else:
            pos.be_trigger = signal.entry_price - sl_dist
            pos.be_sl      = signal.entry_price - fee_per_unit

        self.positions.append(pos)
        logger.info(
            f"[SIM] Opened {pos.direction} @ {pos.entry_price:.2f}  "
            f"SL:{pos.stop_loss:.2f}  TP:{pos.take_profit:.2f}  "
            f"Size:{pos.size:.4f} BTC  ID:{pos.id}"
        )
        return pos

    def update_price(self, price: float) -> Optional[SimPosition]:
        """
        Call on every new price tick.  Checks if any open position
        has hit its TP or SL.  Returns the closed position if one triggers.
        """
        self.current_price = price
        for pos in list(self.positions):
            closed = self._check_exit(pos, price)
            if closed:
                return closed
        self._update_drawdown()
        return None

    def get_open_unrealized_pnl(self) -> float:
        """
        Gross unrealized PnL minus the entry fee already paid and an
        estimated exit fee at the current price.
        """
        if not self.positions or self.current_price == 0:
            return 0.0
        pos = self.positions[0]
        price = self.current_price
        if pos.direction == "LONG":
            gross = (price - pos.entry_price) * pos.size
        else:
            gross = (pos.entry_price - price) * pos.size
        # Deduct the round-trip fee (based on entry margin, charged once at open)
        margin = pos.entry_price * pos.size / pos.leverage
        return gross - margin * config.FEE_TAKER

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_state(self, signal_log: list | None = None) -> None:
        """Persist balance, closed trades, drawdown, and signal log to disk."""
        data = {
            "balance":        self._risk.balance,
            "pos_counter":    self._pos_counter,
            "peak_balance":   self.peak_balance,
            "max_drawdown":   self.max_drawdown,
            "session_wins":   self._risk.session_wins,
            "session_losses": self._risk.session_losses,
            "session_pnl":    self._risk.session_pnl,
            "signal_log":     signal_log or [],
            "closed": [
                {
                    "id":          p.id,
                    "direction":   p.direction,
                    "entry_price": p.entry_price,
                    "stop_loss":   p.stop_loss,
                    "take_profit": p.take_profit,
                    "size":        p.size,
                    "strategy":    p.strategy,
                    "open_time":   p.open_time,
                    "leverage":    p.leverage,
                    "status":      p.status,
                    "close_price": p.close_price,
                    "close_time":  p.close_time,
                    "pnl_usdt":    p.pnl_usdt,
                    "fee_usdt":    p.fee_usdt,
                }
                for p in self.closed
            ],
        }
        try:
            STATE_FILE.write_text(json.dumps(data, indent=2))
            logger.info(
                f"Sim state saved: {len(self.closed)} trades, "
                f"balance=${self._risk.balance:.2f}"
            )
        except Exception as e:
            logger.warning(f"Could not save sim state: {e}")

    def load_state(self) -> list:
        """
        Load persisted state from disk.
        Returns the saved signal_log entries (list of str) so the caller can
        restore them into the dashboard.  Returns [] if no saved state exists.
        """
        if not STATE_FILE.exists():
            return []
        try:
            data = json.loads(STATE_FILE.read_text())

            # Restore risk manager state
            self._risk.balance        = float(data.get("balance",        self._risk.balance))
            self._risk.session_wins   = int(  data.get("session_wins",   0))
            self._risk.session_losses = int(  data.get("session_losses", 0))
            self._risk.session_pnl    = float(data.get("session_pnl",    0.0))

            # Restore sim tracking state
            self._pos_counter = int(  data.get("pos_counter",  0))
            self.peak_balance = float(data.get("peak_balance", self._risk.balance))
            self.max_drawdown = float(data.get("max_drawdown", 0.0))

            # Restore closed positions history
            self.closed = []
            for d in data.get("closed", []):
                p = SimPosition(
                    id          = d["id"],
                    direction   = d["direction"],
                    entry_price = float(d["entry_price"]),
                    stop_loss   = float(d["stop_loss"]),
                    take_profit = float(d["take_profit"]),
                    size        = float(d["size"]),
                    strategy    = d["strategy"],
                    open_time   = float(d["open_time"]),
                    leverage    = float(d["leverage"]),
                    status      = d["status"],
                    close_price = float(d["close_price"]),
                    close_time  = float(d["close_time"]),
                    pnl_usdt    = float(d["pnl_usdt"]),
                    fee_usdt    = float(d["fee_usdt"]),
                )
                self.closed.append(p)

            logger.info(
                f"Sim state loaded: {len(self.closed)} closed trades, "
                f"balance=${self._risk.balance:.2f}  "
                f"({self._risk.session_wins}W / {self._risk.session_losses}L)"
            )
            return data.get("signal_log", [])
        except Exception as e:
            logger.warning(f"Could not load sim state: {e}")
            return []

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        wins   = sum(1 for p in self.closed if p.status == "WIN")
        losses = sum(1 for p in self.closed if p.status == "LOSS")
        be     = sum(1 for p in self.closed if p.status == "BE")
        total  = wins + losses + be
        pnl    = sum(p.pnl_usdt for p in self.closed)
        fees   = sum(p.fee_usdt for p in self.closed)
        return {
            "balance":      self._risk.balance,
            "open_upnl":    self.get_open_unrealized_pnl(),
            "session_pnl":  pnl,
            "total_trades": total,
            "wins":         wins,
            "losses":       losses,
            "be":           be,
            "win_rate":     wins / total if total > 0 else 0.0,
            "max_drawdown": self.max_drawdown,
            "losses_left":  self._risk.losses_remaining(),
            "fees_paid":    fees,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def update_candle(self, candle) -> Optional[SimPosition]:
        """
        Check SL/TP against the full high/low range of a historical candle.
        Used by backtest mode instead of update_price().

        When both SL and TP are hit within the same candle the order is
        inferred from candle direction:
          Bullish (close > open): price went low-first → SL hit first for LONG
          Bearish (close < open): price went high-first → TP hit first for LONG
        """
        if not self.positions:
            return None
        pos = self.positions[0]
        self.current_price = candle.close

        # Slide SL to break-even only if the original SL was NOT also reached
        # in the same candle.  If both trigger and original SL are in range we
        # cannot know intra-candle order; let the existing bullish/bearish logic
        # below handle it so a down-first candle still counts as a real LOSS.
        if not pos.breakeven_set:
            if pos.direction == "LONG":
                if candle.high >= pos.be_trigger and candle.low > pos.stop_loss:
                    pos.stop_loss     = pos.be_sl
                    pos.breakeven_set = True
            else:
                if candle.low <= pos.be_trigger and candle.high < pos.stop_loss:
                    pos.stop_loss     = pos.be_sl
                    pos.breakeven_set = True

        if pos.direction == "LONG":
            hit_sl = candle.low  <= pos.stop_loss
            hit_tp = candle.high >= pos.take_profit
        else:  # SHORT
            hit_sl = candle.high >= pos.stop_loss
            hit_tp = candle.low  <= pos.take_profit

        sl_outcome = "BE" if pos.breakeven_set else "LOSS"

        if hit_sl and hit_tp:
            # Both triggered — infer order from candle direction
            if candle.is_bullish:
                # Low before high: LONG SL first; SHORT TP first → WIN
                if pos.direction == "LONG":
                    return self._close_position(pos, pos.stop_loss, sl_outcome)
                else:
                    return self._close_position(pos, pos.take_profit, "WIN")
            else:
                # High before low: LONG TP first → WIN; SHORT SL first
                if pos.direction == "LONG":
                    return self._close_position(pos, pos.take_profit, "WIN")
                else:
                    return self._close_position(pos, pos.stop_loss, sl_outcome)
        elif hit_tp:
            return self._close_position(pos, pos.take_profit, "WIN")
        elif hit_sl:
            return self._close_position(pos, pos.stop_loss, sl_outcome)

        self._update_drawdown()
        return None

    def _check_exit(self, pos: SimPosition, price: float) -> Optional[SimPosition]:
        # Slide SL to break-even once price reaches the 1:1 point
        if not pos.breakeven_set:
            triggered = (pos.direction == "LONG"  and price >= pos.be_trigger) or \
                        (pos.direction == "SHORT" and price <= pos.be_trigger)
            if triggered:
                pos.stop_loss     = pos.be_sl
                pos.breakeven_set = True
                logger.info(
                    f"[SIM] {pos.id} BE stop activated — SL moved to {pos.be_sl:.2f}"
                )

        hit_sl = hit_tp = False
        if pos.direction == "LONG":
            hit_sl = price <= pos.stop_loss
            hit_tp = price >= pos.take_profit
        else:
            hit_sl = price >= pos.stop_loss
            hit_tp = price <= pos.take_profit

        if hit_tp:
            return self._close_position(pos, price, "WIN")
        if hit_sl:
            # If SL was moved to break-even, closing here is a BE not a LOSS
            outcome = "BE" if pos.breakeven_set else "LOSS"
            return self._close_position(pos, price, outcome)
        return None

    def _close_position(self, pos: SimPosition,
                         exit_price: float, outcome: str) -> SimPosition:
        pos.status      = outcome
        pos.close_price = exit_price
        pos.close_time  = time.time()

        # Gross PnL
        if pos.direction == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * pos.size
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.size

        # Fee = FEE_TAKER × margin (notional / leverage), covers the round trip.
        # FEE_TAKER=0.0004 is the total cost for both open and close legs as a
        # fraction of the margin locked up.  Applying it to notional (× size)
        # would overstate fees by the leverage factor and kill the training signal.
        margin        = pos.entry_price * pos.size / pos.leverage
        total_fee     = margin * config.FEE_TAKER
        net_pnl       = gross_pnl - total_fee

        pos.pnl_usdt  = net_pnl
        pos.fee_usdt  = total_fee

        trade = TradeResult(
            entry_price   = pos.entry_price,
            exit_price    = exit_price,
            direction     = pos.direction,
            size          = pos.size,
            strategy      = pos.strategy,
            timestamp_in  = pos.open_time,
            timestamp_out = pos.close_time,
        )

        if outcome == "WIN":
            self._risk.record_win(net_pnl, trade)
        elif outcome == "LOSS":
            self._risk.record_loss(abs(net_pnl), trade)
        else:
            self._risk.record_breakeven(trade)

        self.positions.remove(pos)
        self.closed.append(pos)

        sign = "+" if net_pnl >= 0 else ""
        logger.info(
            f"[SIM] Closed {pos.id} {outcome}  "
            f"Gross:{'+' if gross_pnl >= 0 else ''}{gross_pnl:.2f}  "
            f"Fee:-{total_fee:.4f}  Net:{sign}{net_pnl:.2f} USDT  "
            f"({pos.strategy})"
        )
        return pos

    def _update_drawdown(self) -> None:
        bal = self._risk.balance
        if bal > self.peak_balance:
            self.peak_balance = bal
        dd = (self.peak_balance - bal) / self.peak_balance if self.peak_balance > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)
