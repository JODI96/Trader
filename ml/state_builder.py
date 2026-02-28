"""
ml/state_builder.py — Converts live indicator snapshots into a fixed 23-float state vector.

State vector layout:
  [0]  vwap_dist_pct / 3.0               → clipped [-1, 1]
  [1]  vwap_zone_encoded                 → {-1, -0.67, -0.33, 0, 0.33, 0.67, 1}
  [2]  htf_trend_encoded                 → {-1, 0, 1}
  [3]  cvd_pct / 50.0                    → clipped [-1, 1]  (session CVD / session vol)
  [4]  cvd_trend_encoded                 → {-1, 0, 1}
  [5]  delta / (avg_delta * 3)           → clipped [-1, 1]
  [6]  buy_sell_ratio - 0.5              → [-0.5, 0.5]
  [7]  signed bubble strength            → [-1, 1]  (+ve=buy bubble, -ve=sell bubble)
  [8]  min(atr_pct / 0.5, 1)            → [0, 1]
  [9]  poc_dist_pct / 2.0               → clipped [-1, 1]
  [10] value_area_position               → {-1, 0..1, 1}
  [11] absorption recency               → [0, 1]  (1.0=just fired, decays to 0 over 20 candles)
  [12] exhaustion recency               → [0, 1]  (1.0=just fired, decays to 0 over 20 candles)
  [13] 3-candle price momentum / 0.5%   → clipped [-1, 1]
  [14] vol_ratio / 3.0                  → clipped [0, 1]
  [15] time_of_day (UTC)                → [0, 1]  (0=midnight, 1=23:59:59)
  [16] weekday (UTC)                    → [0, 1]  (0=Monday, 1=Sunday)
  [17] price_position_20                → [0, 1]  (0=at 20-candle low, 1=at high)
  [18] session                          → {0=Asia, 0.33=London, 0.67=LN/NY overlap, 1=NY}
  [19] atr_trend                        → clipped [-1, 1]  (<0=contracting, >0=expanding)
  [20] support proximity                → [0, 1]  (1.0=price AT support, 0=5+ ATRs away)
  [21] resistance proximity             → [0, 1]  (1.0=price AT resistance, 0=5+ ATRs away)
  [22] vwap_band proximity (signed)     → [-1, 1]  (+1=at upper band, -1=at lower band, 0=far)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from indicators.atr import ATRIndicator
    from indicators.cvd import CVDIndicator
    from indicators.delta_volume import DeltaVolumeIndicator
    from indicators.support_resistance import SupportResistance
    from indicators.volume_profile import VolumeProfile
    from indicators.vwap import VWAPIndicator
    from data.market_data import MarketDataStore

STATE_DIM = 23

_VWAP_ZONE_MAP = {
    "EXTREME_OB": 1.0,
    "STRONG_OB":  0.67,
    "OB":         0.33,
    "NEUTRAL":    0.0,
    "OS":        -0.33,
    "STRONG_OS": -0.67,
    "EXTREME_OS":-1.0,
}

_HTF_TREND_MAP = {
    "BULLISH":  1.0,
    "NEUTRAL":  0.0,
    "BEARISH": -1.0,
}

_CVD_TREND_MAP = {
    "BULLISH":  1.0,
    "NEUTRAL":  0.0,
    "BEARISH": -1.0,
}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_state(
    vwap:         "VWAPIndicator",
    cvd:          "CVDIndicator",
    dv:           "DeltaVolumeIndicator",
    atr:          "ATRIndicator",
    vp:           "VolumeProfile",
    mds:          "MarketDataStore",
    sr:           Optional["SupportResistance"] = None,
    timestamp_ms: int = 0,
) -> List[float]:
    """
    Build the 23-float state vector from current indicator snapshots.
    Always returns a list of exactly STATE_DIM floats (safe defaults for
    indicators still in warmup).

    timestamp_ms: candle bucket start time in milliseconds UTC.  Pass
    candle.ts (open-state) or int(mds.last_trade_ts * 1000) (close-state).
    Defaults to current wall-clock UTC when 0.
    """
    state: List[float] = [0.0] * STATE_DIM

    # [0] VWAP distance — normalised to ±1 at 3% away
    state[0] = _clip(vwap.distance_to_vwap_pct / 3.0, -1.0, 1.0)

    # [1] VWAP zone encoding
    state[1] = _VWAP_ZONE_MAP.get(vwap.zone, 0.0)

    # [2] HTF trend encoding
    state[2] = _HTF_TREND_MAP.get(vwap.htf_trend, 0.0)

    # [3] Session CVD % — CVD as fraction of total session volume.
    # FIX: was using last-10-candle volume as denominator (grew unbounded,
    # always saturated to ±1 after the first hour).  Now uses session total.
    state[3] = _clip(cvd.cumulative_delta_pct / 50.0, -1.0, 1.0)

    # [4] CVD short-term trend
    state[4] = _CVD_TREND_MAP.get(cvd.cvd_trend, 0.0)

    # [5] Per-candle delta vs rolling average
    avg_delta = cvd.avg_delta
    if avg_delta > 0:
        state[5] = _clip(dv.last_delta / (avg_delta * 3.0), -1.0, 1.0)
    else:
        state[5] = 0.0

    # [6] Buy/sell ratio centred around 0
    state[6] = _clip(dv.buy_sell_ratio - 0.5, -0.5, 0.5)

    # [7] Signed bubble strength — FIX: was always positive, direction lost.
    # Now: positive = buy bubble, negative = sell bubble, 0 = no bubble.
    if dv.is_buy_bubble:
        state[7] = min(dv.bubble_strength / 5.0, 1.0)
    elif dv.is_sell_bubble:
        state[7] = -min(dv.bubble_strength / 5.0, 1.0)
    else:
        state[7] = 0.0

    # [8] ATR % of price normalised (1.0 = 0.5%)
    if atr.is_ready:
        state[8] = min(atr.atr_pct / 0.5, 1.0)
    else:
        state[8] = 0.0

    # [9] POC distance normalised
    price = mds.last_price
    if vp.poc > 0 and price > 0:
        poc_dist = (price - vp.poc) / vp.poc * 100.0
        state[9] = _clip(poc_dist / 2.0, -1.0, 1.0)
    else:
        state[9] = 0.0

    # [10] Value area position: -1 = below VAL, +1 = above VAH, 0..1 = inside
    if vp.vah > vp.val and vp.val > 0 and price > 0:
        if price < vp.val:
            state[10] = -1.0
        elif price > vp.vah:
            state[10] = 1.0
        else:
            state[10] = (price - vp.val) / (vp.vah - vp.val)
    else:
        state[10] = 0.0

    # [11] Absorption recency — FIX: was binary 0/1 (fired 1-2% of candles,
    # NN learned to ignore it).  Now decays from 1.0 to 0 over 20 candles.
    state[11] = max(0.0, 1.0 - cvd._candles_since_absorption / 20.0)

    # [12] Exhaustion recency — same fix as [11]
    state[12] = max(0.0, 1.0 - cvd._candles_since_exhaustion / 20.0)

    # [13] 3-candle price momentum normalised to ±1 at ±0.5%
    candles = list(mds.scalp_candles)
    if len(candles) >= 4 and price > 0:
        old_price = candles[-4].close
        momentum_pct = (price - old_price) / old_price * 100.0
        state[13] = _clip(momentum_pct / 0.5, -1.0, 1.0)
    else:
        state[13] = 0.0

    # [14] Volume ratio vs rolling average
    avg_vol = dv.avg_volume
    if avg_vol > 0 and dv.last_volume > 0:
        state[14] = _clip(dv.last_volume / avg_vol / 3.0, 0.0, 1.0)
    else:
        state[14] = 0.0

    # [15] Time of day (UTC) — fraction of 24h elapsed: 0.0=midnight, 1.0=23:59:59
    ts_ms = timestamp_ms if timestamp_ms > 0 else int(datetime.now(timezone.utc).timestamp() * 1000)
    dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    seconds_since_midnight = dt_utc.hour * 3600 + dt_utc.minute * 60 + dt_utc.second
    state[15] = seconds_since_midnight / 86400.0

    # [16] Weekday (UTC) — Monday=0.0, Sunday=1.0
    state[16] = dt_utc.weekday() / 6.0

    # [17] Price position in 20-candle range — 0=at low, 1=at high (RSI-like)
    if len(candles) >= 20 and price > 0:
        hi_20 = max(c.high for c in candles[-20:])
        lo_20 = min(c.low  for c in candles[-20:])
        if hi_20 > lo_20:
            state[17] = (price - lo_20) / (hi_20 - lo_20)
        else:
            state[17] = 0.5
    else:
        state[17] = 0.5

    # [18] Explicit session encoding (UTC):
    #   0.00 = Asia / dead zone  (22:00–08:00)
    #   0.33 = London only       (08:00–13:30)
    #   0.67 = London/NY overlap (13:30–17:00) — highest liquidity
    #   1.00 = NY only           (17:00–22:00)
    hour_frac = dt_utc.hour + dt_utc.minute / 60.0
    if   13.5 <= hour_frac < 17.0:
        state[18] = 0.67
    elif  8.0 <= hour_frac < 13.5:
        state[18] = 0.33
    elif 17.0 <= hour_frac < 22.0:
        state[18] = 1.0
    else:
        state[18] = 0.0

    # [19] ATR trend — compare avg range of last 5 candles vs prior 10 candles
    #   >0 = volatility expanding, <0 = contracting, 0 = stable
    if len(candles) >= 15 and atr.is_ready:
        recent_5  = sum(c.high - c.low for c in candles[-5:])  / 5.0
        prior_10  = sum(c.high - c.low for c in candles[-15:-5]) / 10.0
        if prior_10 > 0:
            state[19] = _clip((recent_5 / prior_10) - 1.0, -1.0, 1.0)
        else:
            state[19] = 0.0
    else:
        state[19] = 0.0

    # [20] Nearest support proximity — 1.0 = price AT support, 0.0 = 5+ ATRs away.
    # High value = strong support floor directly below → bullish context.
    # [21] Nearest resistance proximity — 1.0 = price AT resistance, 0.0 = 5+ ATRs away.
    # High value = ceiling directly above → bearish context.
    if sr is not None and atr.is_ready and atr.atr > 0 and price > 0:
        sup = sr.nearest_support(price)
        res = sr.nearest_resistance(price)
        if sup is not None:
            dist_atrs  = (price - sup) / atr.atr
            state[20]  = max(0.0, 1.0 - dist_atrs / 5.0)
        if res is not None:
            dist_atrs  = (res - price) / atr.atr
            state[21]  = max(0.0, 1.0 - dist_atrs / 5.0)

    # [22] VWAP band proximity (signed).
    # +1.0 = price sitting exactly on an upper band (±1σ/2σ/3σ) → short setup.
    # -1.0 = price sitting exactly on a lower band             → long setup.
    #  0.0 = price is more than 1σ away from any band.
    if vwap.sigma > 0 and price > 0:
        best_dist = float("inf")
        best_sign = 0
        for band, sign in [
            (vwap.upper1, +1), (vwap.upper2, +1), (vwap.upper3, +1),
            (vwap.lower1, -1), (vwap.lower2, -1), (vwap.lower3, -1),
        ]:
            if band > 0:
                d = abs(price - band) / vwap.sigma
                if d < best_dist:
                    best_dist = d
                    best_sign = sign
        state[22] = best_sign * max(0.0, 1.0 - best_dist)

    return state


def compute_confluence(state: List[float], direction: str) -> float:
    """
    Score 0.0–1.0: fraction of indicators aligned with the trade direction.

    Checks 6 independent signals from the state vector:
      HTF trend, VWAP zone, CVD trend, delta pressure,
      absorption/exhaustion signal, volume spike.

    Used to shape the NN reward: high-confluence wins get a bonus,
    low-confluence losses get a bigger penalty.
    """
    is_long = direction == "LONG"
    hits = 0.0
    total = 6.0

    # HTF trend — state[2]: +1=BULLISH, -1=BEARISH
    if (is_long and state[2] > 0) or (not is_long and state[2] < 0):
        hits += 1.0

    # VWAP zone — state[1]: negative=OS (buy area), positive=OB (sell area)
    if (is_long and state[1] < -0.1) or (not is_long and state[1] > 0.1):
        hits += 1.0

    # CVD trend — state[4]: +1=BULLISH, -1=BEARISH
    if (is_long and state[4] > 0) or (not is_long and state[4] < 0):
        hits += 1.0

    # Per-candle delta — state[5]: positive=buy pressure, negative=sell
    if (is_long and state[5] > 0.1) or (not is_long and state[5] < -0.1):
        hits += 1.0

    # Absorption or exhaustion recently fired — state[11] or [12] (recency > 0.5)
    if state[11] > 0.5 or state[12] > 0.5:
        hits += 1.0

    # Volume spike — state[14] > 0.33 means > 1× average
    if state[14] > 0.33:
        hits += 1.0

    return hits / total
