"""
ml/neural_agent.py — Double DQN with LSTM architecture.

Architecture:
    Sequence of N candles × 20 indicator floats
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

Reward function (R-multiples, confluence-shaped in main.py):
    WIN  hitting TP:  ≈ +rr × [0.7 .. 1.3]
    LOSS hitting SL:  ≈ -1.0 × [0.7 .. 1.3]
    Breakeven:        ≈ -0.1 × [0.7 .. 1.3]
    HOLD action:       0.0
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

    REPLAY_MAXLEN       = 25_000
    BATCH_SIZE          = 64
    GAMMA               = 0.95
    LR                  = 3e-4
    EPSILON_START       = 0.8
    EPSILON_MIN         = 0.05
    EPSILON_MIN_AUTO    = 0.10
    EPSILON_DECAY       = 0.9994   # 0.994 decayed in ~440 trades; 0.9994 takes ~5000
    TARGET_UPDATE_EVERY = 20
    SAVE_EVERY          = 10
    BACKUP_EVERY        = 50

    def __init__(self, state_dim: int = 17, n_actions: int = 3,
                 auto_mode: bool = False):
        self._state_dim = state_dim
        self._n_actions = n_actions
        self._auto_mode = auto_mode

        self.epsilon    = self.EPSILON_START
        self._eps_floor = self.EPSILON_MIN_AUTO if auto_mode else self.EPSILON_MIN

        self._steps        = 0
        self._trade_count  = 0
        self._save_count   = 0
        self._total_reward = 0.0
        self._last_loss: Optional[float] = None

        self._recent_actions: Deque[str] = deque(maxlen=5)

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
                    dropout     = 0.1,
                )
                self.head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_actions),
                )

            def forward(self, x):
                # x: (batch, seq_len, state_dim)
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])  # use last timestep

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

        if random.random() < self.epsilon or not self._torch_ok:
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
        """
        if self._seq_prev is None or self._seq_current is None:
            return
        self._buffer.append((self._seq_prev, HOLD, reward, self._seq_current, False))

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
        """
        if self._seq_open is None or self._seq_current is None:
            return

        self._buffer.append((self._seq_open, action, reward, self._seq_current, True))
        self._total_reward += reward
        self._trade_count  += 1

        self.epsilon = max(self._eps_floor, self.epsilon * self.EPSILON_DECAY)

        self._train()

        if self._trade_count % self.TARGET_UPDATE_EVERY == 0 and self._torch_ok:
            self._target.load_state_dict(self._online.state_dict())
            logger.debug("NN target network synced")

        if self._trade_count % self.SAVE_EVERY == 0:
            self.save()

        self._seq_open = None  # clear until next trade opens

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
                # Validate that buffer uses the new MAX_SEQ_LEN format
                if saved and len(saved[0]) == 5:
                    seq = saved[0][0]
                    if (isinstance(seq, list) and len(seq) == MAX_SEQ_LEN
                            and isinstance(seq[0], list) and len(seq[0]) == self._state_dim):
                        self._buffer = deque(saved, maxlen=self.REPLAY_MAXLEN)
                        logger.info(f"NN buffer restored: {len(self._buffer)} experiences")
                    else:
                        logger.info(
                            "NN buffer format changed (LSTM upgrade) — starting fresh"
                        )
            except Exception as e:
                logger.warning(f"NN buffer load failed (starting empty): {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

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

        seq_len = max(1, int(round(config.NN_SEQ_LEN)))
        batch   = random.sample(self._buffer, self.BATCH_SIZE)
        seqs, actions, rewards, next_seqs, dones = zip(*batch)

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
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), 1.0)
        self._optim.step()

        self._last_loss = float(loss.item())
        logger.debug(
            f"NN trained: loss={self._last_loss:.4f}  "
            f"ε={self.epsilon:.3f}  seq={seq_len}"
        )
