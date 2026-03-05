# Trader — Orderflow Scalping Bot

An automated crypto trading bot for Binance Futures that uses a **Double DQN + LSTM neural network** to make trade decisions based on orderflow and market microstructure indicators.

---

## How It Works

### Decision Engine
The bot's only trade decision maker is a neural network. It receives a rolling window of the last N 1-minute candles, each described by a 20-float indicator state vector, and outputs one of three actions: **HOLD**, **LONG**, or **SHORT**.

Architecture: `Sequence (N × 20 floats) → LSTM(128, 2 layers) → Linear(64) → ReLU → Linear(3)`

The model is trained in real-time via **Double DQN** (reinforcement learning). It learns from live or simulated market data — winning trades are rewarded, losing trades are penalized.

### Indicators Used
| Indicator | Description |
|-----------|-------------|
| **ATR** | Average True Range — used for SL/TP distance |
| **VWAP + Bands** | Volume-Weighted Average Price with 1σ/2σ/3σ bands |
| **Volume Profile** | Value Area High/Low, Point of Control |
| **CVD** | Cumulative Volume Delta — buy vs. sell pressure |
| **Delta Volume** | Per-candle delta spikes and absorption signals |
| **Support/Resistance** | Equal highs/lows, liquidity sweeps |
| **Order Book Imbalance** | Top-10-level bid/ask ratio |
| **Multi-timeframe** | 1m + 5m + 15m context |

### Trade Execution
When the NN fires LONG or SHORT and all filters pass:
1. Stop-loss is placed at `NN_SL_ATR_MULT × ATR` from entry
2. Take-profit is automatically set at `SL × MIN_RR_RATIO`
3. Position size is calculated from `RISK_PER_TRADE × balance`
4. After a trade closes, a cooldown of `NN_TRADE_COOLDOWN` candles is enforced

### Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Simulation** | `python main.py --mode sim` | Paper trading — no real orders placed |
| **Alert** | `python main.py --mode alert` | Signals only — you enter manually |
| **Auto** | `python main.py --mode auto` | Live trading on Binance Futures |
| **Backtest** | `python main.py --mode backtest` | Replay historical 1m data |

---

## Installation

### Requirements
- Python 3.10+
- PyTorch (CPU build is sufficient)

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

Install PyTorch separately (CPU version):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Configure environment
```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# true = Testnet (safe), false = Live (real money)
BINANCE_TESTNET=true

SYMBOL=BTCUSDT
INITIAL_BALANCE=1000
LEVERAGE=5
RISK_PER_TRADE=0.005     # 0.5% of balance per trade
MIN_RR_RATIO=3.0          # minimum 1:3 risk/reward
MAX_LOSSES_PER_SESSION=3  # halt after 3 consecutive losses
```

Get your API keys from:
- **Testnet**: https://testnet.binancefuture.com
- **Live**: https://www.binance.com/en/my/settings/api-management

---

## Running the Bot

### Paper trading (recommended to start)
```bash
python main.py --mode sim
```

### Alert mode (signals only, you trade manually)
```bash
python main.py --mode alert --symbol ETHUSDT
```

### Backtest on historical data
```bash
python main.py --mode backtest
```

This downloads the last `BACKTEST_DAYS` (default: 30) days of 1m candles and replays them.

### Live trading
```bash
python main.py --mode auto
```

> **Warning:** Set `BINANCE_TESTNET=false` and use a real API key only when you are confident in the bot's performance.

---

## Downloading Historical Data

To download 1m klines for multiple coins (for backtesting or analysis):

```bash
python fetch_altcoins.py
```

Data is saved to `data/historical/<SYMBOL>/` in CSV format, organized by year and month.

---

## Project Structure

```
Trader/
├── main.py                  # Entry point, RL training loop, WebSocket feed
├── config.py                # All settings loaded from .env
├── fetch_altcoins.py        # Batch historical data downloader
│
├── indicators/              # Technical indicator calculations
│   ├── atr.py
│   ├── cvd.py
│   ├── delta_volume.py
│   ├── support_resistance.py
│   ├── volume_profile.py
│   └── vwap.py
│
├── ml/                      # Machine learning
│   ├── neural_agent.py      # Double DQN + LSTM model, replay buffer, training
│   ├── state_builder.py     # Assembles the 20-float indicator state vector
│   └── param_tuner.py       # Auto-tunes hyperparameters (seq_len, etc.)
│
├── modes/                   # Trading mode implementations
│   ├── simulation.py        # Paper trading
│   ├── alert.py             # Signal-only mode
│   ├── auto_trade.py        # Live order execution
│   ├── backtest.py          # Historical replay
│   └── fetch_data.py        # Kline data fetching helpers
│
├── strategy/                # Trading logic
│   ├── signal_generator.py  # Orderflow signal detection
│   ├── risk_manager.py      # Position sizing, SL/TP calculation
│   └── filters.py           # Volume, news, session filters
│
├── ui/
│   └── dashboard.py         # Rich terminal dashboard
│
├── .env.example             # Environment variable template
└── requirements.txt         # Python dependencies
```

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEVERAGE` | 3 | Futures leverage |
| `RISK_PER_TRADE` | 0.01 | Fraction of balance risked per trade (1%) |
| `MIN_RR_RATIO` | 3.0 | Minimum risk:reward ratio |
| `MAX_LOSSES_PER_SESSION` | 3 | Daily loss circuit-breaker |
| `NN_SL_ATR_MULT` | 1.5 | SL distance as ATR multiple |
| `NN_SEQ_LEN` | 10 | LSTM sequence length (candles) |
| `NN_TRADE_COOLDOWN` | 20 | Candles to wait between trades |
| `BACKTEST_DAYS` | 30 | Days of history for backtesting |

All parameters can be overridden in `.env` without touching the code.
