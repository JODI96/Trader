# Orderflow Scalping Bot

An automated trading system for Binance Futures that combines **market microstructure analysis** with a **Double DQN + LSTM reinforcement learning agent** to generate and execute directional scalping signals on crypto perpetual futures.

---

## Architecture Overview

The system is built in three layers:

1. **Data layer** — WebSocket trade stream + REST kline fetching via `BinanceClient`; a `MarketDataStore` aggregates raw trades into 15-second scalp candles and fans out to higher timeframes (1m, 5m, 15m)
2. **Feature layer** — six real-time indicator modules compute a normalised 28-float state vector each candle
3. **Decision layer** — a Double DQN agent with LSTM backbone reads a rolling window of state vectors and outputs `HOLD`, `LONG`, or `SHORT`; a separate confluence gate requires ≥ 35 % classical-indicator agreement before the NN signal is executed

### Neural Network

```
Sequence (N × 28 floats) → LSTM(hidden=128, layers=2, dropout=0.1)
                         → last hidden state
                         → Linear(128→64) → ReLU
                         → Linear(64→3)   # Q-values: HOLD / LONG / SHORT
```

`N` (sequence length) defaults to 10 candles and is self-tuned by `ParameterTuner` (range 4 – 32).
Each experience in the replay buffer stores `MAX_SEQ_LEN = 32` states (zero-padded) so past experiences remain valid when the tuner changes `N`.

**Training regime:**
Training runs exclusively in `BACKTEST` mode via high-speed historical replay — no live capital is required. In `SIMULATION` and `AUTO` modes the network runs in inference-only mode (weights frozen).

### State Vector — 28 Features

| # | Feature | Range |
|---|---------|-------|
| 0 | VWAP distance (% / 3) | [-1, 1] |
| 1 | VWAP zone (6 bands) | {-1 … 1} |
| 2 | 15m HTF trend | {-1, 0, 1} |
| 3 | Session CVD % | [-1, 1] |
| 4 | CVD trend | {-1, 0, 1} |
| 5 | Per-candle delta / (3 × avg delta) | [-1, 1] |
| 6 | Buy/sell ratio − 0.5 | [-0.5, 0.5] |
| 7 | Signed bubble strength | [-1, 1] |
| 8 | ATR % of price | [0, 1] |
| 9 | POC distance % | [-1, 1] |
| 10 | Value-area position | [-1, 1] |
| 11 | Absorption recency (20-candle decay) | [0, 1] |
| 12 | Exhaustion recency (20-candle decay) | [0, 1] |
| 13 | 3-candle price momentum | [-1, 1] |
| 14 | Volume ratio / 3 | [0, 1] |
| 15 | Time of day (UTC) | [0, 1] |
| 16 | Weekday (UTC) | [0, 1] |
| 17 | Price position within 20-candle range | [0, 1] |
| 18 | Session (Asia / London / Overlap / NY) | {0, 0.33, 0.67, 1} |
| 19 | ATR trend (expanding / contracting) | [-1, 1] |
| 20 | Support proximity | [0, 1] |
| 21 | Resistance proximity | [0, 1] |
| 22 | VWAP band proximity (signed) | [-1, 1] |
| 23 | Order-book imbalance (top-10 levels) | [-1, 1] |
| 24 | 1m EMA(5)/EMA(10) cross | {-1, 0, 1} |
| 25 | 1m 3-candle momentum | [-1, 1] |
| 26 | 5m EMA(3)/EMA(6) cross | {-1, 0, 1} |
| 27 | 5m 3-candle momentum | [-1, 1] |

### Reward Design

Reward shaping is intentionally asymmetric to keep the model honest at low win rates:

| Outcome | Base reward | Shaped by |
|---------|-------------|-----------|
| TP hit (WIN) | +10.0 | × [0.7 – 1.3] × confluence |
| Break-even stop (BE) | +2.0 | flat |
| SL hit (LOSS) | −4.0 | × [1.3 – 0.7] × confluence |
| HOLD — no setup | +0.05 | flat |
| HOLD — clear setup missed | −0.20 | flat |

**Intra-trade shaping** accumulates per-candle while a position is open (capped at ±15) and is added to the outcome reward at close, clamped by sign:

- `SpeedBonus`: +3 · exp(−0.3t) when price moves in the correct direction
- `VolBonus`: +3 · exp(−0.3t) when volume accelerates in the correct direction
- `VolWrongPen`: −3 when volume surges against the open direction
- `ConsolPen`: −0.2 per candle when price progress < 0.5 × ATR

Break-even win rate (WIN vs. LOSS only): 4 / (10 + 4) ≈ 28.6 %.

**HOLD subsampling:** HOLD experiences are stored at 1/20 frequency to prevent the replay buffer from being dominated by reward-zero experiences, which would collapse Q-values.

### Indicators

| Module | Signals |
|--------|---------|
| `ATR` | 14-period Average True Range; used for SL/TP sizing |
| `VWAP` | Session VWAP with 1σ / 2σ / 3σ bands; 15m HTF trend |
| `VolumeProfile` | Value-area high/low, Point of Control |
| `CVD` | Cumulative Volume Delta; absorption and exhaustion detection |
| `DeltaVolume` | Per-candle delta, buy/sell ratio, bubble detection |
| `SupportResistance` | Equal highs/lows, liquidity sweep levels (scalp + HTF) |

Order-book imbalance (top-10 bid/ask levels) is polled via REST alongside the WebSocket stream.

### Trade Execution

When the NN outputs LONG or SHORT and all conditions pass:

1. SL placed at `NN_SL_ATR_MULT × ATR` from entry
2. TP set at `SL distance × MIN_RR_RATIO`
3. Position size computed from `RISK_PER_TRADE × current balance`
4. `NN_TRADE_COOLDOWN` candles are enforced between trades

### Operating Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Simulation** | `python main.py --mode sim` | Paper trading with full P&L tracking |
| **Alert** | `python main.py --mode alert` | Signal-only output; no orders placed |
| **Auto** | `python main.py --mode auto` | Live order execution on Binance Futures |
| **Backtest** | `python main.py --mode backtest` | Historical replay with RL training |
| **Validate** | `python main.py --mode validate` | Inference-only replay (weights frozen) |
| **Fetch** | `python main.py --mode fetch` | Download and cache historical kline data |

---

## Installation

### Requirements

- Python 3.10+
- PyTorch (CPU build is sufficient for inference and backtest)

### 1. Clone the repo

```bash
git clone https://github.com/JODI96/Trader.git
cd Trader
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch separately (CPU build):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# true = Testnet  |  false = Live
BINANCE_TESTNET=true

SYMBOL=BTCUSDT
INITIAL_BALANCE=1000
LEVERAGE=5
RISK_PER_TRADE=0.005      # 0.5 % of balance per trade
MIN_RR_RATIO=3.0           # minimum 1:3 risk / reward
MAX_LOSSES_PER_SESSION=3   # circuit-breaker: halt after N consecutive losses
```

API keys:
- Testnet: https://testnet.binancefuture.com
- Live: https://www.binance.com/en/my/settings/api-management

---

## Usage

### Paper trading

```bash
python main.py --mode sim
```

### Signal alerts (manual execution)

```bash
python main.py --mode alert --symbol ETHUSDT
```

### Backtest and train on historical data

```bash
# Download last 30 days, replay and train
python main.py --mode backtest

# Specific month from local dataset
python main.py --mode backtest --year 2025 --month 01

# Multiple cycles
python main.py --mode backtest --cycles 5 --year 2025
```

### Validate without updating weights

```bash
python main.py --mode validate --year 2025
```

### Download historical data

```bash
python main.py --mode fetch --year 2025 --month 03
# or use the bulk downloader:
python fetch_altcoins.py
```

Data is saved to `data/historical/<SYMBOL>/` in CSV format, organised by year and month.

### Live trading

```bash
python main.py --mode auto
```

> **Warning:** Set `BINANCE_TESTNET=false` and use a real API key only after thorough backtesting and simulation.

### Model utilities

```bash
# Re-enable exploration (e.g. after extended backtest)
python main.py --set-epsilon 0.5

# Remove oldest N experiences from replay buffer
python main.py --trim-buffer 5000

# Flush experiences recorded during low-win-rate periods
python main.py --flush-bad
```

---

## Project Structure

```
Trader/
├── main.py                    # Entry point, RL loop, WebSocket orchestration
├── config.py                  # All settings loaded from .env
├── fetch_altcoins.py          # Batch historical data downloader
│
├── data/                      # Market data access
│   ├── binance_client.py      # WebSocket stream, REST klines, OBI polling
│   ├── economic_calendar.py   # High-impact news event filter
│   ├── historical_store.py    # Local CSV dataset management
│   └── market_data.py         # MarketDataStore — trade aggregation, candle fan-out
│
├── indicators/                # Real-time indicator calculations
│   ├── atr.py
│   ├── cvd.py
│   ├── delta_volume.py
│   ├── support_resistance.py
│   ├── volume_profile.py
│   └── vwap.py
│
├── ml/                        # Reinforcement learning
│   ├── neural_agent.py        # Double DQN + LSTM, prioritised replay buffer
│   ├── state_builder.py       # 28-float indicator → state vector
│   └── param_tuner.py         # Adaptive hyperparameter tuning (seq_len, etc.)
│
├── modes/                     # Runtime mode implementations
│   ├── simulation.py          # Paper trading with persistent state
│   ├── alert.py               # Terminal signal output
│   ├── auto_trade.py          # Live order management
│   ├── backtest.py            # High-speed historical replay
│   └── fetch_data.py          # Kline download helpers
│
├── strategy/                  # Signal and risk logic
│   ├── signal_generator.py    # Orderflow signal detection
│   ├── risk_manager.py        # Position sizing, SL/TP, session circuit-breaker
│   └── filters.py             # Volume, news, session filters
│
├── ui/
│   └── dashboard.py           # Rich terminal dashboard with live chart
│
├── .env.example               # Environment variable template
└── requirements.txt
```

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEVERAGE` | 3 | Futures leverage |
| `RISK_PER_TRADE` | 0.01 | Fraction of balance per trade |
| `MIN_RR_RATIO` | 3.0 | Minimum risk : reward ratio |
| `MAX_LOSSES_PER_SESSION` | 3 | Consecutive-loss circuit-breaker |
| `NN_SL_ATR_MULT` | 1.5 | SL distance as ATR multiple |
| `NN_SEQ_LEN` | 10 | LSTM input sequence length (candles) |
| `NN_TRADE_COOLDOWN` | 20 | Candles between trades |
| `BACKTEST_DAYS` | 30 | Default history window for backtesting |
| `MTF_5M_LOOKBACK` | 60 | 5m candles prefetched at startup |

All parameters can be overridden in `.env` without modifying source code.
