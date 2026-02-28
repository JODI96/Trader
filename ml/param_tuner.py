"""
ml/param_tuner.py — Self-tuning indicator parameters via policy gradient.

For each tunable parameter:
  - At trade open: apply small Gaussian perturbation to the parameter.
  - At trade close: record (noise, shaped_reward) pair.
  - Every UPDATE_EVERY trades: nudge each parameter in the direction that
    correlated with positive reward (simplified evolution-strategies gradient).

Tunable parameters and their search ranges:
  DELTA_BUBBLE_MULT   [1.0, 4.0]  — bubble detection sensitivity
  ABSORPTION_BODY     [0.10, 0.60] — candle body ratio for absorption signal
  LOW_VOLUME_FACTOR   [0.20, 0.80] — low-volume filter threshold (× avg)
  IMBALANCE_RATIO     [1.5, 6.0]  — buy/sell volume ratio for imbalance play
  SWEEP_REVERSAL_PCT  [0.01, 0.10] — snap-back % for liquidity sweep confirm
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

PARAMS_PATH = Path(__file__).parent / "model" / "params.json"

# (config_attr, default, min, max, noise_sigma)
PARAM_SPECS: List[Tuple[str, float, float, float, float]] = [
    ("DELTA_BUBBLE_MULT",  2.0,  1.0,  4.0,  0.05),
    ("ABSORPTION_BODY",    0.30, 0.10, 0.60, 0.02),
    ("LOW_VOLUME_FACTOR",  0.50, 0.20, 0.80, 0.02),
    ("IMBALANCE_RATIO",    3.0,  1.5,  6.0,  0.10),
    ("SWEEP_REVERSAL_PCT", 0.03, 0.01, 0.10, 0.005),
    # LSTM sequence length: how many consecutive candles the NN analyses.
    # Stored as float for gradient math, applied as int to config.
    ("NN_SEQ_LEN",         10.0, 4.0,  32.0, 2.0),
]

# Params that must be applied as int (rounded) to config
_INT_PARAMS = {"NN_SEQ_LEN"}

PARAM_DEFAULTS: Dict[str, float] = {name: default for name, default, *_ in PARAM_SPECS}


class ParameterTuner:
    """
    Adjusts indicator sensitivity parameters over time using a simplified
    policy gradient (evolution-strategies estimate):

        theta_new = theta + alpha * (1/N) * sum(noise_i * reward_i)

    Parameters start at their config defaults, then drift toward values that
    produce better shaped trade outcomes.
    """

    UPDATE_EVERY = 30   # trades between parameter updates
    ALPHA        = 0.02  # step size for parameter updates

    def __init__(self) -> None:
        # Current (un-perturbed) parameter values
        self._values: Dict[str, float] = dict(PARAM_DEFAULTS)

        # Ring buffer of (noise_dict, shaped_reward) for gradient estimate
        self._history: List[Tuple[Dict[str, float], float]] = []

        # Noise applied to the current open trade (cleared when trade closes)
        self._pending_noise: Optional[Dict[str, float]] = None

        self._trade_count = 0

        self.load()
        self._apply_to_config()

    # ── Public API ──────────────────────────────────────────────────────────

    def perturb(self) -> None:
        """
        Sample small Gaussian noise and temporarily apply it to config.
        Call this when a trade is about to open.
        """
        noise: Dict[str, float] = {}
        for name, default, lo, hi, sigma in PARAM_SPECS:
            n = random.gauss(0.0, sigma)
            noise[name] = n
            perturbed = max(lo, min(hi, self._values[name] + n))
            applied = int(round(perturbed)) if name in _INT_PARAMS else perturbed
            setattr(config, name, applied)
        self._pending_noise = noise

    def record(self, shaped_reward: float) -> None:
        """
        Record the outcome of the perturbed trade and trigger a parameter
        update every UPDATE_EVERY trades.
        Call this when a trade closes.
        """
        if self._pending_noise is None:
            return

        self._history.append((self._pending_noise, shaped_reward))
        self._pending_noise = None
        self._trade_count += 1

        if self._trade_count % self.UPDATE_EVERY == 0:
            self._update_params()
            self.save()

    def current_values(self) -> Dict[str, float]:
        """Snapshot of current (un-perturbed) parameter values."""
        return dict(self._values)

    def save(self) -> None:
        """Persist current parameter values to JSON."""
        try:
            PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
            PARAMS_PATH.write_text(json.dumps({
                "values":      self._values,
                "trade_count": self._trade_count,
            }, indent=2))
            logger.debug(f"Params saved after {self._trade_count} trades: {self._values}")
        except Exception as e:
            logger.warning(f"Params save failed: {e}")

    def load(self) -> None:
        """Load persisted parameter values if available."""
        if not PARAMS_PATH.exists():
            return
        try:
            data = json.loads(PARAMS_PATH.read_text())
            loaded = data.get("values", {})
            for name, default, lo, hi, _ in PARAM_SPECS:
                if name in loaded:
                    self._values[name] = max(lo, min(hi, float(loaded[name])))
            self._trade_count = int(data.get("trade_count", 0))
            logger.info(f"Params loaded ({self._trade_count} trades): {self._values}")
        except Exception as e:
            logger.warning(f"Params load failed (using defaults): {e}")

    # ── Internal ────────────────────────────────────────────────────────────

    def _update_params(self) -> None:
        """
        Policy gradient update using the most recent UPDATE_EVERY samples:
            delta_theta_i = alpha * mean(noise_i * reward)
        """
        window = self._history[-self.UPDATE_EVERY:]
        n = len(window)
        if n == 0:
            return

        for name, default, lo, hi, _ in PARAM_SPECS:
            gradient = sum(noise[name] * rew for noise, rew in window) / n
            old_val = self._values[name]
            new_val = max(lo, min(hi, old_val + self.ALPHA * gradient))
            self._values[name] = new_val
            if abs(new_val - old_val) > 1e-4:
                logger.info(
                    f"Param tuned: {name}  {old_val:.4f} -> {new_val:.4f}  "
                    f"(grad={gradient:+.4f})"
                )

        self._apply_to_config()

    def _apply_to_config(self) -> None:
        """Write current (un-perturbed) values back to the config module."""
        for name, val in self._values.items():
            applied = int(round(val)) if name in _INT_PARAMS else val
            setattr(config, name, applied)
