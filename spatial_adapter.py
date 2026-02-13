# spatial_adapter.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TrialResult:
    fixated: bool
    t_fix: float
    t: np.ndarray
    H: np.ndarray   # Shannon entropy over time (global composition)
    p: np.ndarray   # composition over time, shape (T, 3)

def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    p: (..., 3) composition fractions summing to 1.
    returns entropy in nats
    """
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)

def run_spatial_trial(
    epsilon: float,
    sigma_noise: float,
    seed: int,
    *,
    L: int = 60,
    T_max: int = 50_000,
    sample_every: int = 50,
) -> TrialResult:
    """
    TODO: Replace the body with your real spatial simulator call.

    Must return:
      - fixated: True if any strategy reaches 1.0 global fraction before T_max
      - t_fix: time (in steps) when fixation occurs, else T_max
      - t: sampled time array
      - H: entropy series at sampled times
      - p: sampled composition series (T_samples, 3)
    """
    rng = np.random.default_rng(seed)

    # --- Placeholder toy evolution (REPLACE) ---
    # Start near uniform
    p = np.array([1/3, 1/3, 1/3], dtype=float)
    t_list, p_list = [], []

    fixated = False
    t_fix = float(T_max)

    for step in range(T_max + 1):
        # Toy stochastic drift with bias magnitude ~ epsilon and noise ~ sigma_noise
        dp = rng.normal(0.0, sigma_noise, size=3)
        dp += epsilon * np.array([0.5, -0.25, -0.25])
        p = p + 1e-3 * dp
        p = np.clip(p, 0.0, None)
        s = p.sum()
        p = p / s if s > 0 else np.array([1/3, 1/3, 1/3])

        if step % sample_every == 0:
            t_list.append(step)
            p_list.append(p.copy())

        if (p.max() > 1.0 - 1e-6) and (not fixated):
            fixated = True
            t_fix = float(step)
            break

    t_arr = np.array(t_list, dtype=float)
    p_arr = np.array(p_list, dtype=float)
    H_arr = shannon_entropy(p_arr)
    return TrialResult(fixated=fixated, t_fix=t_fix, t=t_arr, H=H_arr, p=p_arr)

