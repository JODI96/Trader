"""
Central configuration — loaded from .env file.
All modules import from here; never read os.getenv() directly elsewhere.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Binance API ───────────────────────────────────────────────────────────────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET    = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# ─── Symbol ────────────────────────────────────────────────────────────────────
SYMBOL             = os.getenv("SYMBOL", "BTCUSDT")

# ─── Account ───────────────────────────────────────────────────────────────────
INITIAL_BALANCE    = float(os.getenv("INITIAL_BALANCE", "10000"))
LEVERAGE           = int(os.getenv("LEVERAGE", "3"))
RISK_PER_TRADE     = float(os.getenv("RISK_PER_TRADE", "0.01"))   # 1%
MIN_RR_RATIO       = float(os.getenv("MIN_RR_RATIO", "3.0"))
MAX_LOSSES         = int(os.getenv("MAX_LOSSES_PER_SESSION", "3"))

# ─── Fees ──────────────────────────────────────────────────────────────────────
# Round-trip fee applied to the margin (notional / leverage) used by the position.
# 0.04% of margin covers both the open and close legs.
# Applied as: total_fee = (entry_price × size / leverage) × FEE_TAKER
# Applying to raw notional (without /leverage) would overstate fees by the
# leverage factor, making tight ATR stops unviable for training.
FEE_TAKER          = float(os.getenv("FEE_TAKER", "0.0004"))  # 0.04% of margin, round-trip

# ─── Timeframes (seconds) ──────────────────────────────────────────────────────
SCALP_TF_SEC       = 15      # 15-second scalping candles
HTF_TF_SEC         = 900     # 15-minute context candles

# ─── Indicator Parameters ──────────────────────────────────────────────────────
ATR_PERIOD         = 14
VWAP_BANDS         = [1.0, 2.0, 3.0]        # sigma multipliers
VP_BINS            = 50                       # volume profile price buckets
VP_VALUE_AREA_PCT  = 0.70                    # 70% of volume = value area
LOOKBACK_CANDLES   = 200                     # 15s candles kept in memory
HTF_LOOKBACK       = 100                     # 15min candles kept

# ─── Strategy Thresholds ───────────────────────────────────────────────────────
DELTA_BUBBLE_MULT  = 2.0     # delta > 2× rolling avg → bubble
IMBALANCE_RATIO    = 3.0     # 3:1 buy/sell volume ratio
ABSORPTION_BODY    = 0.30    # candle body < 30% of ATR = absorption
EQUAL_HL_TOL_PCT   = 0.05    # 0.05% tolerance for equal H/L detection
SWEEP_REVERSAL_PCT = 0.03    # price must snap back 0.03% for sweep confirmation

# ─── Filters ───────────────────────────────────────────────────────────────────
LOW_VOLUME_FACTOR  = 0.5     # < 50% of 20-bar avg vol = skip
NEWS_BUFFER_MIN    = 15      # minutes before/after high-impact news
LOW_LIQUIDITY_UTC  = (12, 14)  # UTC hour range to avoid (12:00–14:00)

# ─── Alerts ────────────────────────────────────────────────────────────────────
ALERT_BEEP_COUNT   = int(os.getenv("ALERT_BEEP_COUNT", "3"))

# ─── Neural Network ────────────────────────────────────────────────────────────
# The NN is the sole trade decision maker.
# It reads a 15-float state vector built from all indicators and outputs
# HOLD / LONG / SHORT.  When LONG or SHORT fires and all filters pass,
# an ATR-based SL/TP is calculated and the trade is opened.

# SL distance = NN_SL_ATR_MULT × ATR  (TP = SL × MIN_RR_RATIO automatically)
NN_SL_ATR_MULT     = float(os.getenv("NN_SL_ATR_MULT", "1.5"))

# How many consecutive candles the LSTM sees as a sequence.
# Tuned automatically by ParameterTuner (range 4–32).
NN_SEQ_LEN         = int(os.getenv("NN_SEQ_LEN", "10"))

# Minimum candles to wait after a trade closes before opening another.
# Prevents back-to-back trades that burn fees without giving the NN time to
# observe the market. Default 5 candles (= 5 min on 1m backtest, 75s live).
NN_TRADE_COOLDOWN  = int(os.getenv("NN_TRADE_COOLDOWN", "10"))

# Leverage used during backtest — lower than live to reduce fee burn rate.
# The NN reward is in R-multiples so it doesn't change what the NN learns,
# but the paper account lasts 2.5x longer giving more training trades.
BACKTEST_LEVERAGE  = int(os.getenv("BACKTEST_LEVERAGE", "3"))

# In SIMULATION mode the session loss limit is disabled so the NN can keep
# collecting experiences all day without being halted after 3 losses.
# In ALERT / AUTO mode the limit is always enforced regardless of this flag.
SIM_NO_LOSS_LIMIT  = os.getenv("SIM_NO_LOSS_LIMIT", "true").lower() == "true"

# ─── Backtesting ───────────────────────────────────────────────────────────────
# Number of days of 1m historical klines to download when running --mode backtest.
BACKTEST_DAYS      = int(os.getenv("BACKTEST_DAYS", "30"))
