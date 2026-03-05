"""
ml/neural_agent.py — Double DQN with LSTM architecture.

Architecture:
    Sequence of N 1-minute candles × 20 indicator floats
    → LSTM(input=20, hidden=128, num_layers=2, dropout=0.1)
    → last hidden state
    → Linear(64) → ReLU
    → Linear(3)   # Q-values: HOLD=0, LONG=1, SHORT=2

The LSTM sees a rolling window of the last N candles' full indicator states,
so it learns temporal multi-indicator patterns — e.g., "CVD building for
several candles while volume compresses, then a delta spike fires near VWAP."

Sequence length N is tunable by ParameterTuner (config.NN_SEQ_LEN, default 10,
range 4–32).  Each replay-buffer experience stores MAX_SEQ_LEN=32 states
(zero-padded at the start when the buffer is not yet full).  During training
the sequences are sliced to the current config.NN_SEQ_LEN so historical
experiences remain valid when the tuner changes N.

Actions:
    0 = HOLD
    1 = LONG
    2 = SHORT

Reward function (shaped in main.py):
    WIN  hitting TP:  +10.0 × [0.7..1.3 conf] + max(0, shaping)  (backtest: +10 + max(0,s))
    LOSS hitting SL:   -4.0 × [1.3..0.7 conf] + min(0, shaping)  (backtest:  -4 + min(0,s))
    Breakeven:         -0.5 + min(0, shaping)
    HOLD action:  +0.05 (conf<0.35) | 0.0 (0.35–0.60) | -0.20 (conf>0.60, missed setup)
    Break-even win rate: 4/(10+4) ≈ 28.6%  (above random, below HOLD-collapse threshold)
    Shaping is CLAMPED by outcome — positive shaping never rescues a loss,
    negative shaping never penalises a win.
    intra_shaping (per candle while in trade, accumulated):
        SpeedBonus  +3·exp(-0.3t)  if price_progress>0 AND price_velocity>0
        VolBonus    +3·exp(-0.3t)  if vol_accel>0.3   AND price_progress>0
        VolWrongPen -3             if vol_accel>0.3   AND price_progress<0
        ConsolPen   -0.2           if |price_progress| < 0.5×ATR  (suppressed on strong moves)
        cap: ±15
"""
from __future__ import annotations

import logging
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# Action constants
HOLD  = 0
LONG  = 1
SHORT = 2
ACTION_NAMES = {HOLD: "HOLD", LONG: "LONG", SHORT: "SHORT"}

MODEL_DIR   = Path(__file__).parent / "model"
MODEL_PATH  = MODEL_DIR / "agent.pt"
BUFFER_PATH = MODEL_DIR / "buffer.pkl"

# Sequences in the replay buffer are always this long (zero-padded if needed).
# Training slices them to config.NN_SEQ_LEN so old experiences stay valid
# when the sequence length is tuned.
MAX_SEQ_LEN = 32


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class DQNAgent:
    """
    Double DQN agent with LSTM sequence encoder.

    Public API (identical to the old feedforward version — main.py needs only
    one extra call: snapshot_open_seq() when a trade is confirmed open):

        agent.select_action(state)        -> int
        agent.snapshot_open_seq()          # call immediately when trade opens
        agent.record_hold(state, next_state)
        agent.record_trade(state, action, reward, next_state)
        agent.stats()                      -> dict
        agent.save() / agent.load()
    """

    REPLAY_MAXLEN       = 100_000   # was 30k — more diverse experience history
    BATCH_SIZE          = 128      # was 64  — smoother gradient estimates
    GAMMA               = 0.95
    LR                  = 1e-3# was 1e-3 — less aggressive updates, less drift
    EPSILON_START       = 0.8
    EPSILON_MIN         = 0.05     # never fully stop exploring (prevents policy lock-in)
    EPSILON_MIN_AUTO    = 0.02
    EPSILON_DECAY       = 0.9997   # slower: hits floor at ~13k trades vs ~5.5k at 0.9995
    TARGET_UPDATE_EVERY  = 50       # was 20  — more stable Q targets
    SAVE_EVERY           = 10
    BACKUP_EVERY         = 50

    def __init__(self, state_dim: int = 17, n_actions: int = 3,
                 auto_mode: bool = False):
        self._state_dim = state_dim
        self._n_actions = n_actions
        self._auto_mode = auto_mode

        self.epsilon    = self.EPSILON_START
        self._eps_floor = self.EPSILON_MIN_AUTO if auto_mode else self.EPSILON_MIN
        self._training  = True   # False → pure inference: no exploration, no learning

        self._steps        = 0
        self._trade_count  = 0
        self._wins         = 0   # trades with reward > 0 (used for bad-flag tagging)
        self._save_count   = 0
        self._total_reward = 0.0
        self._last_loss: Optional[float] = None

        self._recent_actions: Deque[str] = deque(maxlen=5)
        self._recent_rewards: Deque[float] = deque(maxlen=100)  # rolling window for bad-flag

        # Replay buffer — each entry: (max_seq, action, reward, max_next_seq, done)
        self._buffer: Deque[Tuple] = deque(maxlen=self.REPLAY_MAXLEN)

        # Rolling state history — one 15-float snapshot per candle
        self._state_buffer: Deque[List[float]] = deque(maxlen=MAX_SEQ_LEN)

        # Sequence snapshots updated on every select_action call
        self._seq_prev:    Optional[List[List[float]]] = None  # candle t-1 seq
        self._seq_current: Optional[List[List[float]]] = None  # candle t seq
        self._seq_open:    Optional[List[List[float]]] = None  # seq at trade open

        self._torch_ok = _torch_available()
        if not self._torch_ok:
            logger.warning(
                "torch not found — NN agent will use random actions only. "
                "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._build_networks()
        self.load()

    # ── Network construction ──────────────────────────────────────────────────

    def _build_networks(self) -> None:
        import torch
        import torch.nn as nn

        state_dim = self._state_dim
        n_actions = self._n_actions

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size  = state_dim,
                    hidden_size = 128,
                    num_layers  = 2,
                    batch_first = True,
                    dropout     = 0.2,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, n_actions),
                )

            def forward(self, x):
                # x: (batch, seq_len, state_dim)
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        self._torch  = torch
        self._online = LSTMNet()
        self._target = LSTMNet()
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optim   = torch.optim.Adam(self._online.parameters(), lr=self.LR)
        self._loss_fn = torch.nn.SmoothL1Loss()

    # ── Public API ────────────────────────────────────────────────────────────

    def select_action(self, state: List[float]) -> int:
        """
        Epsilon-greedy action selection using the LSTM over the last
        seq_len candles.  Appends the new state to the rolling buffer
        and updates the internal sequence snapshots used by record_hold /
        record_trade.
        """
        self._steps += 1
        self._state_buffer.append(state)

        # Shift snapshots: prev ← current, current ← new full seq
        self._seq_prev    = self._seq_current
        self._seq_current = self._build_max_seq()

        if self._training and (random.random() < self.epsilon or not self._torch_ok):
            action = random.randint(0, self._n_actions - 1)
        elif not self._torch_ok:
            action = random.randint(0, self._n_actions - 1)
        else:
            import torch
            seq_len = max(1, int(round(config.NN_SEQ_LEN)))
            seq     = self._build_seq(seq_len)
            with torch.no_grad():
                t = torch.tensor([seq], dtype=torch.float32)  # (1, seq_len, 15)
                q = self._online(t)
                action = int(q.argmax(dim=1).item())

        self._recent_actions.append(ACTION_NAMES[action][0])
        return action

    def snapshot_open_seq(self) -> None:
        """
        Save the current sequence as the open-state for the trade that is
        about to start.  Call this in main.py the moment a trade is confirmed
        open (before record_trade is called at close).
        """
        self._seq_open = self._seq_current

    def record_hold(self, state: List[float], next_state: List[float],
                    reward: float = 0.0) -> None:
        """
        Record a HOLD experience using the internally tracked sequence snapshots.
        reward=0.0 by default; pass a small positive value (+0.04) when
        confluence is low to teach the NN that avoiding bad setups has value.
        The state arguments are accepted for API compatibility but the sequences
        stored internally are used.
        No-op when training is disabled.
        """
        if not self._training:
            return
        if self._seq_prev is None or self._seq_current is None:
            return
        bad = self._win_rate() < 0.5
        self._buffer.append((self._seq_prev, HOLD, reward, self._seq_current, False, bad))

        # Train every 10 HOLD records so the model learns from between-trade states,
        # not just trade closes. 338 trades/month → ~560 updates/month (+66% frequency).
        if self._steps % 10 == 0:
            self._train()

        # Periodic buffer flush to disk
        if len(self._buffer) % 100 == 0 and self._torch_ok:
            self._save_buffer()

    def record_trade(self, state: List[float], action: int,
                     reward: float, next_state: List[float]) -> None:
        """
        Record a completed trade and trigger training.
        Uses _seq_open (saved at trade open) and _seq_current (at close).
        The state/next_state arguments are accepted for API compatibility.

        reward: confluence-shaped R-multiple (computed in main.py)
        No-op when training is disabled (validate / sim / live mode).
        """
        if not self._training:
            # Still count trades and reward for stats, but don't learn
            self._total_reward += reward
            self._trade_count  += 1
            return

        if self._seq_open is None or self._seq_current is None:
            return

        # Tag with bad flag BEFORE updating counts (reflects performance up to now)
        bad = self._win_rate() < 0.5
        self._buffer.append((self._seq_open, action, reward, self._seq_current, True, bad))
        self._total_reward += reward
        self._trade_count  += 1
        if reward > 0:
            self._wins += 1
        self._recent_rewards.append(1.0 if reward > 0 else 0.0)

        self.epsilon = max(self._eps_floor, self.epsilon * self.EPSILON_DECAY)

        self._train()

        if self._trade_count % self.TARGET_UPDATE_EVERY == 0 and self._torch_ok:
            self._target.load_state_dict(self._online.state_dict())
            logger.debug("NN target network synced")

        if self._trade_count % self.SAVE_EVERY == 0:
            self.save()

        self._seq_open = None  # clear until next trade opens

    def trim_buffer(self, n: int) -> int:
        """
        Remove the oldest `n` experiences from the replay buffer and save.
        Returns how many were actually removed.
        """
        n = min(n, len(self._buffer))
        for _ in range(n):
            self._buffer.popleft()
        if n > 0:
            self._save_buffer()
            logger.info(f"Buffer trimmed: removed {n} oldest experiences, {len(self._buffer)} remain")
        return n

    def set_training(self, enabled: bool) -> None:
        """
        Enable or disable learning.

        enabled=True  (default) — training mode: explores, records experiences,
                                  updates weights, decays epsilon.
        enabled=False           — inference mode: always greedy, no buffer writes,
                                  no weight updates (validate / sim / live).
        """
        self._training = enabled
        if not enabled:
            self.epsilon = 0.0   # pure exploitation

    def stats(self) -> dict:
        recent  = "".join(self._recent_actions) or "-"
        seq_len = int(round(getattr(config, "NN_SEQ_LEN", 10)))
        buf     = len(self._buffer)
        cap     = self.REPLAY_MAXLEN
        # Epsilon progress toward floor (0% = just started, 100% = at floor)
        eps_range = self.EPSILON_START - self._eps_floor
        eps_progress = max(0.0, min(1.0,
            (self.EPSILON_START - self.epsilon) / eps_range if eps_range > 0 else 0.0
        ))
        return {
            "training":     self._training,
            "epsilon":      round(self.epsilon, 3),
            "eps_floor":    self._eps_floor,
            "eps_pct":      round(eps_progress * 100, 1),   # % toward floor
            "steps":        self._steps,
            "total_reward": round(self._total_reward, 2),
            "last_loss":    round(self._last_loss, 4) if self._last_loss is not None else None,
            "recent":       recent,
            "trade_count":  self._trade_count,
            "seq_len":      seq_len,
            "buf_size":     buf,
            "buf_capacity": cap,
            "buf_pct":      round(buf / cap * 100, 1),      # % buffer filled
            "lr":           self.LR,
            "state_dim":    self._state_dim,
        }

    def _save_buffer(self) -> None:
        """Atomic buffer save — writes to .tmp then renames to prevent corruption."""
        tmp = BUFFER_PATH.with_suffix(".tmp")
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                pickle.dump(list(self._buffer), f)
            tmp.replace(BUFFER_PATH)
        except Exception as e:
            logger.warning(f"NN buffer save failed: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def save(self) -> None:
        if not self._torch_ok:
            return
        self._save_count += 1
        data = {
            "state_dict":   self._online.state_dict(),
            "epsilon":      self.epsilon,
            "steps":        self._steps,
            "trade_count":  self._trade_count,
            "wins":         self._wins,
            "total_reward": self._total_reward,
        }
        self._torch.save(data, MODEL_PATH)

        self._save_buffer()

        if self._save_count % self.BACKUP_EVERY == 0:
            backup = MODEL_DIR / f"backup_{self._save_count}.pt"
            self._torch.save(data, backup)
            logger.info(f"NN backup saved: {backup}")
        else:
            logger.debug(
                f"NN saved: trade #{self._trade_count}  "
                f"buf={len(self._buffer)}  seq_len={int(round(config.NN_SEQ_LEN))}"
            )

    def load(self) -> None:
        if not self._torch_ok or not MODEL_PATH.exists():
            return
        try:
            data = self._torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            self._online.load_state_dict(data["state_dict"])
            self._target.load_state_dict(data["state_dict"])
            self.epsilon       = max(self._eps_floor, float(data.get("epsilon", self.EPSILON_START)))
            self._steps        = int(data.get("steps", 0))
            self._trade_count  = int(data.get("trade_count", 0))
            self._wins         = int(data.get("wins", 0))
            self._total_reward = float(data.get("total_reward", 0.0))
            logger.info(
                f"NN LSTM model loaded: ε={self.epsilon:.3f}  "
                f"trades={self._trade_count}  totalR={self._total_reward:.2f}"
            )
        except Exception as e:
            logger.warning(f"NN model load failed (starting fresh): {e}")

        if BUFFER_PATH.exists():
            try:
                with open(BUFFER_PATH, "rb") as f:
                    saved = pickle.load(f)
                # Accept 5-element (old) and 6-element (new, with bad flag) formats
                if saved and len(saved[0]) in (5, 6):
                    seq = saved[0][0]
                    if (isinstance(seq, list) and len(seq) == MAX_SEQ_LEN
                            and isinstance(seq[0], list) and len(seq[0]) == self._state_dim):
                        # Upgrade old 5-element entries to 6-element (bad=False)
                        upgraded = [e if len(e) == 6 else (*e, False) for e in saved]
                        self._buffer = deque(upgraded, maxlen=self.REPLAY_MAXLEN)
                        logger.info(f"NN buffer restored: {len(self._buffer)} experiences")
                    else:
                        logger.info(
                            "NN buffer format changed (LSTM upgrade) — starting fresh"
                        )
            except Exception as e:
                logger.warning(f"NN buffer load failed (starting empty): {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _win_rate(self) -> float:
        """Rolling win rate over the last 100 trades (0.5 until enough data)."""
        if not self._recent_rewards:
            return 0.5
        return sum(self._recent_rewards) / len(self._recent_rewards)

    def flush_bad(self) -> int:
        """
        Remove all experiences tagged as bad (recorded when win rate < 50%).
        Saves the pruned buffer to disk. Returns the number removed.
        """
        before = len(self._buffer)
        good   = deque(
            (e for e in self._buffer if not e[5]),
            maxlen=self.REPLAY_MAXLEN,
        )
        removed = before - len(good)
        self._buffer = good
        if removed > 0:
            self._save_buffer()
            logger.info(f"flush_bad: removed {removed:,} bad experiences, {len(self._buffer):,} remain")
        return removed

    def _build_seq(self, seq_len: int) -> List[List[float]]:
        """
        Return a list of exactly seq_len state vectors from the rolling buffer,
        zero-padded at the start when not enough history is available yet.
        """
        buf = list(self._state_buffer)
        if len(buf) >= seq_len:
            return buf[-seq_len:]
        padding = [[0.0] * self._state_dim] * (seq_len - len(buf))
        return padding + buf

    def _build_max_seq(self) -> List[List[float]]:
        """Full MAX_SEQ_LEN sequence for storage in the replay buffer."""
        return self._build_seq(MAX_SEQ_LEN)

    def _train(self) -> None:
        if not self._torch_ok or len(self._buffer) < self.BATCH_SIZE:
            if len(self._buffer) < self.BATCH_SIZE:
                logger.debug(
                    f"NN buffer warming up: {len(self._buffer)}/{self.BATCH_SIZE}"
                )
            return

        import torch

        seq_len  = max(1, int(round(config.NN_SEQ_LEN)))
        batch    = random.sample(self._buffer, self.BATCH_SIZE)
        unpacked = list(zip(*batch))
        seqs, actions, rewards, next_seqs, dones = unpacked[:5]  # ignore bad flag

        def to_tensor(seq_batch):
            # Slice each MAX_SEQ_LEN sequence to current seq_len (last N steps)
            sliced = [s[-seq_len:] for s in seq_batch]
            return torch.tensor(sliced, dtype=torch.float32)  # (batch, seq_len, 15)

        states_t      = to_tensor(seqs)
        next_states_t = to_tensor(next_seqs)
        actions_t     = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32)
        dones_t       = torch.tensor(dones,   dtype=torch.float32)

        # Current Q-values from online network
        current_q = self._online(states_t).gather(1, actions_t).squeeze(1)

        # Double DQN target: online selects action, target evaluates it
        with torch.no_grad():
            online_next  = self._online(next_states_t)
            best_actions = online_next.argmax(dim=1, keepdim=True)
            target_next  = self._target(next_states_t).gather(1, best_actions).squeeze(1)
            target_q     = rewards_t + self.GAMMA * target_next * (1.0 - dones_t)

        loss = self._loss_fn(current_q, target_q)
        self._optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), 0.5)  # was 1.0
        self._optim.step()

        self._last_loss = float(loss.item())
        logger.debug(
            f"NN trained: loss={self._last_loss:.4f}  "
            f"ε={self.epsilon:.3f}  seq={seq_len}"
        )
