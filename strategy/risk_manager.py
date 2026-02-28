"""
risk_manager.py — Position sizing, R:R validation, session loss tracking.

Strict rules:
  - Risk per trade: RISK_PER_TRADE × account balance
  - Minimum R:R 1:3
  - Maximum 3 stop-losses per session → trading halted after 3rd
  - No martingale, no scaling into losers
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    entry_price:  float
    exit_price:   float
    direction:    str       # 'LONG' | 'SHORT'
    size:         float     # contracts/base units
    strategy:     str
    timestamp_in:  float
    timestamp_out: float = 0.0
    outcome:       str    = ""   # 'WIN' | 'LOSS' | 'BE'
    pnl_usdt:      float  = 0.0


class RiskManager:
    """
    Manages session-level risk state and computes trade parameters.
    """

    def __init__(self, initial_balance: float = config.INITIAL_BALANCE,
                 sim_mode: bool = False):
        self._initial_balance: float = initial_balance
        self.balance:      float = initial_balance
        self.session_losses: int = 0
        self.session_wins:   int = 0
        self.session_pnl:   float = 0.0
        self._session_start: float = time.time()
        self._trade_history: list[TradeResult] = []
        self._sim_mode: bool = sim_mode

    # ── Session control ─────────────────────────────────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        """
        Returns (True, "") if trading is allowed, or (False, reason) if not.
        In SIMULATION mode the session loss limit is skipped so the NN can
        keep learning all day (controlled by config.SIM_NO_LOSS_LIMIT).
        """
        if self._sim_mode and config.SIM_NO_LOSS_LIMIT:
            if self.balance < self._initial_balance * 0.05:
                return False, "Balance depleted"
            return True, ""
        if self.session_losses >= config.MAX_LOSSES:
            return False, f"Session loss limit reached ({config.MAX_LOSSES} losses)"
        return True, ""

    def losses_remaining(self) -> int:
        if self._sim_mode and config.SIM_NO_LOSS_LIMIT:
            return 999   # unlimited in sim learning mode
        return max(0, config.MAX_LOSSES - self.session_losses)

    def reset_session(self) -> None:
        """Call at start of new trading session (e.g. daily reset)."""
        self.session_losses  = 0
        self.session_wins    = 0
        self.session_pnl     = 0.0
        self._session_start  = time.time()
        logger.info("Session reset — loss counter cleared")

    # ── Position sizing ─────────────────────────────────────────────────────

    def calculate_size(self, entry: float, stop_loss: float) -> float:
        """
        Position size in base units (BTC).
        size = (balance × risk_pct) / |entry - stop_loss|

        Leverage is NOT applied here — it is implicit in futures margin
        (the exchange lets you hold the notional with less cash), but the
        dollar risk on the trade is always balance × RISK_PER_TRADE.
        Multiplying by leverage here would create positions 5× too large and
        make fees dwarf any edge.
        """
        sl_distance = abs(entry - stop_loss)
        if sl_distance == 0:
            logger.warning("SL distance is 0 — cannot size position")
            return 0.0
        risk_usdt = self.balance * config.RISK_PER_TRADE
        size      = risk_usdt / sl_distance
        # Round to 3 decimal places (Binance minimum step for BTCUSDT)
        size = round(size, 3)
        return max(size, 0.001)

    def calculate_take_profit(self, entry: float, stop_loss: float,
                               direction: str) -> float:
        """
        TP at minimum 1:RR_RATIO distance on the profit side.
        """
        sl_distance = abs(entry - stop_loss)
        tp_distance = sl_distance * config.MIN_RR_RATIO
        if direction == "LONG":
            return entry + tp_distance
        return entry - tp_distance

    def check_rr(self, entry: float, stop_loss: float,
                  take_profit: float, direction: str) -> tuple[bool, float]:
        """
        Validates R:R ratio.
        Returns (is_valid, actual_rr_ratio).
        """
        sl_dist = abs(entry - stop_loss)
        if direction == "LONG":
            tp_dist = take_profit - entry
        else:
            tp_dist = entry - take_profit

        if sl_dist == 0:
            return False, 0.0
        rr = tp_dist / sl_dist
        return rr >= config.MIN_RR_RATIO, rr

    # ── Trade recording ─────────────────────────────────────────────────────

    def record_win(self, pnl: float, trade: Optional[TradeResult] = None) -> None:
        self.session_wins += 1
        self.session_pnl  += pnl
        self.balance      += pnl
        if trade:
            trade.outcome   = "WIN"
            trade.pnl_usdt  = pnl
            self._trade_history.append(trade)
        logger.info(f"WIN recorded: +${pnl:.2f}  |  balance: ${self.balance:.2f}")

    def record_loss(self, loss: float, trade: Optional[TradeResult] = None) -> None:
        """loss should be a positive number (the amount lost)."""
        self.session_losses += 1
        self.session_pnl    -= loss
        self.balance        -= loss
        if trade:
            trade.outcome   = "LOSS"
            trade.pnl_usdt  = -loss
            self._trade_history.append(trade)
        remaining = self.losses_remaining()
        logger.warning(
            f"LOSS recorded: -${loss:.2f}  |  balance: ${self.balance:.2f}  |  "
            f"losses remaining: {remaining}/{config.MAX_LOSSES}"
        )
        if remaining == 0:
            logger.warning("SESSION HALTED — maximum losses reached!")

    def record_breakeven(self, trade: Optional[TradeResult] = None) -> None:
        if trade:
            trade.outcome  = "BE"
            trade.pnl_usdt = 0.0
            self._trade_history.append(trade)
        logger.info("Breakeven recorded")

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        return self.session_wins + self.session_losses

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.session_wins / self.total_trades

    @property
    def session_duration_min(self) -> float:
        return (time.time() - self._session_start) / 60.0

    def summary(self) -> dict:
        return {
            "balance":        self.balance,
            "session_pnl":    self.session_pnl,
            "trades":         self.total_trades,
            "wins":           self.session_wins,
            "losses":         self.session_losses,
            "win_rate":       self.win_rate,
            "losses_left":    self.losses_remaining(),
            "duration_min":   self.session_duration_min,
        }
