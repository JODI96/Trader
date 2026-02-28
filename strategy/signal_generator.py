"""
signal_generator.py — Six orderflow scalping strategies.

Each strategy method returns a Signal or None.
The main `check()` method runs all strategies and returns the highest-confidence
signal (or None if no A+ setup is present).

Strategies:
  1. Absorption         — high volume, no price progress → counter trade
  2. Delta Divergence   — price new H/L with CVD divergence
  3. Imbalance Play     — 3:1 buy/sell imbalance cluster + pullback
  4. Liquidity Sweep    — equal H/L swept + snap back
  5. VWAP Bounce        — HTF trend + pullback to VWAP/±1σ + delta confirm
  6. Breakout Pullback  — VAH/VAL break with volume + pullback
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import config
from data.market_data import Candle, Signal
from indicators.atr import ATRIndicator
from indicators.cvd import CVDIndicator
from indicators.delta_volume import DeltaVolumeIndicator
from indicators.support_resistance import SupportResistance
from indicators.volume_profile import VolumeProfile
from indicators.vwap import VWAPIndicator

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Evaluates all 6 strategies on each new candle and emits Signal objects.
    """

    def __init__(
        self,
        atr:   ATRIndicator,
        dv:    DeltaVolumeIndicator,
        vwap:  VWAPIndicator,
        cvd:   CVDIndicator,
        vp:    VolumeProfile,
        sr:    SupportResistance,
    ):
        self._atr  = atr
        self._dv   = dv
        self._vwap = vwap
        self._cvd  = cvd
        self._vp   = vp
        self._sr   = sr

        self._recent_candles: List[Candle] = []
        self._cooldown_until: float = 0.0   # prevent signal spam
        self.COOLDOWN_SEC = 30              # 2 candles worth of cooldown

    # ── Public API ──────────────────────────────────────────────────────────

    def update_candle(self, candle: Candle) -> None:
        self._recent_candles.append(candle)
        if len(self._recent_candles) > 50:
            self._recent_candles.pop(0)

    def check(self) -> Optional[Signal]:
        """
        Run all 6 strategies.  Return the highest-confidence signal,
        or None if no valid setup exists.
        """
        if not self._atr.is_ready or len(self._recent_candles) < 10:
            return None
        if time.time() < self._cooldown_until:
            return None

        signals: List[Signal] = []
        for strategy_fn in [
            self._absorption,
            self._delta_divergence,
            self._imbalance_play,
            self._liquidity_sweep,
            self._vwap_bounce,
            self._breakout_pullback,
        ]:
            try:
                sig = strategy_fn()
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"Strategy error in {strategy_fn.__name__}: {e}")

        if not signals:
            return None

        # Return highest confidence signal; break ties by R:R
        best = max(signals, key=lambda s: (s.confidence, s.rr_ratio))
        self._cooldown_until = time.time() + self.COOLDOWN_SEC
        logger.info(
            f"Signal: {best.direction} {best.strategy} "
            f"@ {best.entry_price:.2f}  Conf:{best.confidence}/3"
        )
        return best

    # ── Strategy 1: Absorption ───────────────────────────────────────────────

    def _absorption(self) -> Optional[Signal]:
        """
        Heavy delta but tiny candle body → passive orders absorbing flow.
        Buy absorption (heavy buys, no up move)  → SHORT
        Sell absorption (heavy sells, no down move) → LONG
        """
        if not self._cvd.absorption_detected:
            return None

        direction = self._cvd.absorption_direction
        if not direction:
            return None

        # Require delta bubble confirmation
        if not self._dv.is_any_bubble:
            return None

        candle  = self._recent_candles[-1]
        atr     = self._atr.atr
        price   = candle.close

        if direction == "SHORT":
            entry = price
            sl    = candle.high + atr * 0.3
            tp    = self._calc_tp(entry, sl, "SHORT")
        else:
            entry = price
            sl    = candle.low - atr * 0.3
            tp    = self._calc_tp(entry, sl, "LONG")

        rr_ok, rr = self._check_rr(entry, sl, tp, direction)
        if not rr_ok:
            return None

        confidence = 1
        if self._dv.bubble_strength > config.DELTA_BUBBLE_MULT * 1.5:
            confidence += 1
        if self._vwap.zone in ("OB", "STRONG_OB") and direction == "SHORT":
            confidence += 1
        if self._vwap.zone in ("OS", "STRONG_OS") and direction == "LONG":
            confidence += 1
        confidence = min(confidence, 3)

        return Signal(
            timestamp   = time.time(),
            direction   = direction,
            strategy    = "Absorption",
            entry_price = entry,
            stop_loss   = sl,
            take_profit = tp,
            rr_ratio    = rr,
            confidence  = confidence,
            reason      = (
                f"Delta {self._dv.bubble_strength:.1f}× avg, "
                f"body {candle.body_size:.2f} vs ATR {atr:.2f}"
            ),
        )

    # ── Strategy 2: Delta Divergence ─────────────────────────────────────────

    def _delta_divergence(self) -> Optional[Signal]:
        """
        Price new high but CVD lower → SHORT
        Price new low but CVD higher → LONG
        """
        if not self._cvd.exhaustion_detected:
            return None

        direction = self._cvd.exhaustion_direction
        if not direction:
            return None

        candle = self._recent_candles[-1]
        atr    = self._atr.atr
        price  = candle.close

        if direction == "SHORT":
            entry = price
            sl    = candle.high + atr * 0.5
            tp    = self._calc_tp(entry, sl, "SHORT")
        else:
            entry = price
            sl    = candle.low - atr * 0.5
            tp    = self._calc_tp(entry, sl, "LONG")

        rr_ok, rr = self._check_rr(entry, sl, tp, direction)
        if not rr_ok:
            return None

        # Confirm with VWAP zone
        confidence = 1
        if direction == "SHORT" and "OB" in self._vwap.zone:
            confidence += 1
        if direction == "LONG"  and "OS" in self._vwap.zone:
            confidence += 1
        # Near HTF resistance for shorts, support for longs
        if direction == "SHORT":
            res = self._sr.nearest_resistance(price)
            if res and abs(res - price) / price < 0.001:
                confidence += 1
        if direction == "LONG":
            sup = self._sr.nearest_support(price)
            if sup and abs(sup - price) / price < 0.001:
                confidence += 1
        confidence = min(confidence, 3)

        return Signal(
            timestamp   = time.time(),
            direction   = direction,
            strategy    = "Delta Divergence",
            entry_price = entry,
            stop_loss   = sl,
            take_profit = tp,
            rr_ratio    = rr,
            confidence  = confidence,
            reason      = (
                f"Price new {'high' if direction == 'SHORT' else 'low'} "
                f"but CVD diverging ({self._cvd.cvd_trend})"
            ),
        )

    # ── Strategy 3: Imbalance Play ───────────────────────────────────────────

    def _imbalance_play(self) -> Optional[Signal]:
        """
        3:1 buy/sell volume imbalance at a price level.
        Wait for pullback into imbalance zone + delta continuation.
        """
        if len(self._recent_candles) < 5:
            return None

        # Look for imbalance in last 3 candles
        for lookback in range(1, 4):
            c = self._recent_candles[-lookback]
            if c.volume == 0:
                continue
            buy_ratio  = c.buy_vol  / c.volume
            sell_ratio = c.sell_vol / c.volume

            if buy_ratio >= config.IMBALANCE_RATIO / (config.IMBALANCE_RATIO + 1):
                # Strong buy imbalance → bullish
                direction = "LONG"
                imbalance_price = c.close
            elif sell_ratio >= config.IMBALANCE_RATIO / (config.IMBALANCE_RATIO + 1):
                # Strong sell imbalance → bearish
                direction = "SHORT"
                imbalance_price = c.close
            else:
                continue

            # Current price must have pulled back into the imbalance zone
            current = self._recent_candles[-1].close
            atr     = self._atr.atr

            if direction == "LONG" and current <= imbalance_price + atr * 0.5:
                # Delta must be turning bullish
                if self._dv.delta_trend != "BULLISH":
                    continue
                entry = current
                sl    = current - atr * 1.0
                tp    = self._calc_tp(entry, sl, "LONG")
                rr_ok, rr = self._check_rr(entry, sl, tp, "LONG")
                if not rr_ok:
                    continue

                confidence = 1
                if self._vwap.zone in ("OS", "NEUTRAL"):
                    confidence += 1
                if self._cvd.cvd_trend == "BULLISH":
                    confidence += 1

                return Signal(
                    timestamp=time.time(), direction="LONG",
                    strategy="Imbalance Play",
                    entry_price=entry, stop_loss=sl, take_profit=tp,
                    rr_ratio=rr, confidence=min(confidence, 3),
                    reason=f"Buy imbalance {buy_ratio:.0%} at {imbalance_price:.2f}",
                )

            if direction == "SHORT" and current >= imbalance_price - atr * 0.5:
                if self._dv.delta_trend != "BEARISH":
                    continue
                entry = current
                sl    = current + atr * 1.0
                tp    = self._calc_tp(entry, sl, "SHORT")
                rr_ok, rr = self._check_rr(entry, sl, tp, "SHORT")
                if not rr_ok:
                    continue

                confidence = 1
                if self._vwap.zone in ("OB", "NEUTRAL"):
                    confidence += 1
                if self._cvd.cvd_trend == "BEARISH":
                    confidence += 1

                return Signal(
                    timestamp=time.time(), direction="SHORT",
                    strategy="Imbalance Play",
                    entry_price=entry, stop_loss=sl, take_profit=tp,
                    rr_ratio=rr, confidence=min(confidence, 3),
                    reason=f"Sell imbalance {sell_ratio:.0%} at {imbalance_price:.2f}",
                )

        return None

    # ── Strategy 4: Liquidity Sweep ──────────────────────────────────────────

    def _liquidity_sweep(self) -> Optional[Signal]:
        """
        Equal highs / lows get swept with high volume, then snap back.
        Enter opposite direction of sweep.
        """
        candle = self._recent_candles[-1]

        swept_high = self._sr.swept_equal_high(candle)
        swept_low  = self._sr.swept_equal_low(candle)

        if not swept_high and not swept_low:
            return None

        atr = self._atr.atr

        # Confirm volume spike
        if not self._dv.is_any_bubble and self._dv.last_volume < self._dv.avg_volume * 1.3:
            return None

        if swept_high:
            direction = "SHORT"
            entry = candle.close
            sl    = candle.high + atr * 0.2   # just above the wick
            tp    = self._calc_tp(entry, sl, "SHORT")
        else:
            direction = "LONG"
            entry = candle.close
            sl    = candle.low - atr * 0.2    # just below the wick
            tp    = self._calc_tp(entry, sl, "LONG")

        rr_ok, rr = self._check_rr(entry, sl, tp, direction)
        if not rr_ok:
            return None

        # Snap-back confirmation: price must have closed back inside the range
        snap_pct = abs(candle.close - (swept_high or swept_low)) / (swept_high or swept_low)
        if snap_pct < config.SWEEP_REVERSAL_PCT / 100.0:
            return None

        confidence = 2   # sweep by definition is strong
        if self._dv.is_any_bubble:
            confidence = 3

        level = swept_high or swept_low
        return Signal(
            timestamp   = time.time(),
            direction   = direction,
            strategy    = "Liquidity Sweep",
            entry_price = entry,
            stop_loss   = sl,
            take_profit = tp,
            rr_ratio    = rr,
            confidence  = confidence,
            reason      = f"Equal {'high' if swept_high else 'low'} swept @ {level:.2f}",
        )

    # ── Strategy 5: VWAP Bounce ──────────────────────────────────────────────

    def _vwap_bounce(self) -> Optional[Signal]:
        """
        HTF trend + pullback to VWAP or ±1σ + delta confirms reversal.
        """
        if self._vwap.vwap == 0:
            return None

        htf_trend = self._vwap.htf_trend
        if htf_trend == "NEUTRAL":
            return None

        direction: Optional[str] = None

        # LONG: HTF bullish + price pulled back to VWAP or lower1
        if htf_trend == "BULLISH":
            if self._vwap.is_at_vwap(0.4) or self._vwap.is_at_band(1.0) == "LOWER":
                if self._dv.delta_trend in ("BULLISH", "NEUTRAL"):
                    direction = "LONG"

        # SHORT: HTF bearish + price pulled back up to VWAP or upper1
        if htf_trend == "BEARISH":
            if self._vwap.is_at_vwap(0.4) or self._vwap.is_at_band(1.0) == "UPPER":
                if self._dv.delta_trend in ("BEARISH", "NEUTRAL"):
                    direction = "SHORT"

        if direction is None:
            return None

        candle = self._recent_candles[-1]
        atr    = self._atr.atr
        price  = candle.close

        if direction == "LONG":
            entry = price
            sl    = min(candle.low, self._vwap.lower1) - atr * 0.3
            tp    = self._calc_tp(entry, sl, "LONG")
        else:
            entry = price
            sl    = max(candle.high, self._vwap.upper1) + atr * 0.3
            tp    = self._calc_tp(entry, sl, "SHORT")

        rr_ok, rr = self._check_rr(entry, sl, tp, direction)
        if not rr_ok:
            return None

        confidence = 1
        if self._cvd.cvd_trend == ("BULLISH" if direction == "LONG" else "BEARISH"):
            confidence += 1
        if self._sr.is_near_level(price, 0.1):
            confidence += 1

        return Signal(
            timestamp   = time.time(),
            direction   = direction,
            strategy    = "VWAP Bounce",
            entry_price = entry,
            stop_loss   = sl,
            take_profit = tp,
            rr_ratio    = rr,
            confidence  = confidence,
            reason      = (
                f"HTF {htf_trend} + price at VWAP {self._vwap.vwap:.2f} "
                f"(dist {self._vwap.distance_to_vwap_pct:.2f}%)"
            ),
        )

    # ── Strategy 6: Breakout Pullback ────────────────────────────────────────

    def _breakout_pullback(self) -> Optional[Signal]:
        """
        Break of VAH/VAL with strong volume → wait for pullback to broken level.
        """
        if self._vp.vah == 0 or self._vp.val == 0:
            return None
        if len(self._recent_candles) < 5:
            return None

        # Find the breakout candle (within last 5 bars)
        vah = self._vp.vah
        val = self._vp.val
        atr = self._atr.atr

        for i in range(-5, -1):
            c = self._recent_candles[i]
            breakout_dir = None

            if c.close > vah and c.volume > self._dv.avg_volume * 1.5:
                breakout_dir = "LONG"
            elif c.close < val and c.volume > self._dv.avg_volume * 1.5:
                breakout_dir = "SHORT"

            if breakout_dir is None:
                continue

            current = self._recent_candles[-1].close

            # Price must have pulled back toward broken level
            if breakout_dir == "LONG":
                if not (val <= current <= vah + atr * 0.5):
                    continue
                if self._dv.delta_trend == "BEARISH":
                    continue   # delta still bearish = no continuation
                entry = current
                sl    = current - atr * 1.0
                tp    = self._calc_tp(entry, sl, "LONG")
            else:
                if not (val - atr * 0.5 <= current <= vah):
                    continue
                if self._dv.delta_trend == "BULLISH":
                    continue
                entry = current
                sl    = current + atr * 1.0
                tp    = self._calc_tp(entry, sl, "SHORT")

            rr_ok, rr = self._check_rr(entry, sl, tp, breakout_dir)
            if not rr_ok:
                continue

            confidence = 1
            if c.volume > self._dv.avg_volume * 2.0:
                confidence += 1
            if self._cvd.cvd_trend == ("BULLISH" if breakout_dir == "LONG" else "BEARISH"):
                confidence += 1

            level = vah if breakout_dir == "LONG" else val
            return Signal(
                timestamp   = time.time(),
                direction   = breakout_dir,
                strategy    = "Breakout Pullback",
                entry_price = entry,
                stop_loss   = sl,
                take_profit = tp,
                rr_ratio    = rr,
                confidence  = confidence,
                reason      = f"{'VAH' if breakout_dir == 'LONG' else 'VAL'} breakout pullback @ {level:.2f}",
            )

        return None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _calc_tp(self, entry: float, sl: float, direction: str) -> float:
        dist = abs(entry - sl)
        tp_dist = dist * config.MIN_RR_RATIO
        return entry + tp_dist if direction == "LONG" else entry - tp_dist

    def _check_rr(self, entry: float, sl: float,
                   tp: float, direction: str) -> tuple[bool, float]:
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            return False, 0.0
        tp_dist = (tp - entry) if direction == "LONG" else (entry - tp)
        rr = tp_dist / sl_dist
        return rr >= config.MIN_RR_RATIO, round(rr, 2)
