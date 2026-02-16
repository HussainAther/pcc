#!/usr/bin/env python3
"""
spatial_adapter.py

Spatial stochastic cyclic dominance (RPS) lattice simulator.

Key design goal:
  - Provide a spatial rule that actually produces "dissipative" drift toward fixation,
    analogous to the mean-field dissipative replicator instability.

Implementation:
  - L x L lattice, periodic BC
  - Moore neighborhood
  - Random sequential updates
  - 1 MCS = `attempts_per_mcs` interaction attempts (default = L^2)
  - Maintain global counts so global fractions are cheap to access
  - Invasion probability depends on global frequency difference between winner & loser:
        p_win = clip(base_p + eps_gain * epsilon * (p_w - p_l), 0, 1)

This is a minimal spatial analogue of replicator-like amplification:
  - When a strategy is more abundant, its successful invasions become slightly more likely.
  - This creates drift away from coexistence and toward fixation as epsilon increases.

Outputs:
  TrialResult(fixated, t_fix, t, H, p)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TrialResult:
    fixated: bool
    t_fix: float
    t: np.ndarray      # sampled times (MCS)
    H: np.ndarray      # Shannon entropy at sampled times
    p: np.ndarray      # fractions (n_samples, 3)


def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        q = np.clip(p, eps, 1.0)
        q = q / q.sum()
        return -np.sum(q * np.log(q))
    q = np.clip(p, eps, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    return -np.sum(q * np.log(q), axis=1)


def beats(a: int, b: int) -> bool:
    """
    Cyclic dominance:
      0 beats 1
      1 beats 2
      2 beats 0
    """
    return (a - b) % 3 == 2


# Moore neighborhood offsets
NEIGHBORS = np.array(
    [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ],
    dtype=np.int32
)


def run_spatial_trial(
    epsilon: float,
    sigma_noise: float,
    seed: int,
    *,
    L: int = 100,
    T_max: int = 50_000,           # in MCS
    sample_every: int = 50,        # in MCS
    base_p: float = 0.5,
    eps_gain: float = 6.0,
    attempts_per_mcs: int | None = None,
    fixation_tol: float = 1e-6,
) -> TrialResult:
    """
    Run one spatial stochastic trial.

    Args:
      epsilon: destabilizing strength (controls drift away from coexistence)
      sigma_noise: per-interaction noise added to p_win (optional)
      seed: RNG seed
      L: lattice size
      T_max: max simulation time in MCS
      sample_every: sampling interval in MCS
      base_p: baseline invasion probability (0.5 is neutral)
      eps_gain: amplifies epsilon effect; larger => faster fixation
      attempts_per_mcs: number of interaction attempts per MCS (default L^2).
                        For fast tests you can set e.g. L^2//5.
      fixation_tol: fixation detection tolerance on fractions

    Returns:
      TrialResult
    """
    rng = np.random.default_rng(int(seed))

    # Initialize lattice randomly
    grid = rng.integers(0, 3, size=(L, L), dtype=np.int8)

    # Maintain counts for O(1) global fractions
    counts = np.bincount(grid.ravel(), minlength=3).astype(np.int64)
    N = int(L * L)

    if attempts_per_mcs is None:
        attempts_per_mcs = N
    attempts_per_mcs = int(max(1, attempts_per_mcs))

    # Sampling buffers
    max_samples = (T_max // sample_every) + 2
    t_list = np.zeros(max_samples, dtype=float)
    p_list = np.zeros((max_samples, 3), dtype=float)
    s = 0

    fixated = False
    t_fix = float(T_max)

    def fracs() -> np.ndarray:
        return counts.astype(float) / float(N)

    for mcs in range(T_max + 1):

        # sample
        if (mcs % sample_every) == 0:
            p = fracs()
            t_list[s] = float(mcs)
            p_list[s] = p
            s += 1

            if p.max() >= (1.0 - fixation_tol):
                fixated = True
                t_fix = float(mcs)
                break

        if mcs == T_max:
            break

        # interaction attempts
        for _ in range(attempts_per_mcs):
            i = rng.integers(L)
            j = rng.integers(L)
            di, dj = NEIGHBORS[rng.integers(8)]
            ni = (i + di) % L
            nj = (j + dj) % L

            a = int(grid[i, j])
            b = int(grid[ni, nj])

            if a == b:
                continue

            # Determine winner/loser by cyclic dominance
            if beats(a, b):
                winner, loser = a, b
                win_site = (ni, nj)   # loser site becomes winner
            elif beats(b, a):
                winner, loser = b, a
                win_site = (i, j)     # loser site becomes winner
            else:
                continue

            # Replicator-like dissipative drift:
            # invasion success is biased by global frequency difference (p_w - p_l)
            p_global = counts.astype(float) / float(N)
            delta = float(p_global[winner] - p_global[loser])

            p_win = base_p + float(eps_gain) * float(epsilon) * delta

            if sigma_noise > 0.0:
                p_win += float(rng.normal(0.0, sigma_noise))

            p_win = float(np.clip(p_win, 0.0, 1.0))

            if rng.random() < p_win:
                # perform replacement at loser site
                wi, wj = win_site
                old = int(grid[wi, wj])
                if old != winner:
                    grid[wi, wj] = np.int8(winner)
                    counts[old] -= 1
                    counts[winner] += 1

    # trim samples
    t = t_list[:s].copy()
    p = p_list[:s].copy()
    H = shannon_entropy(p)

    return TrialResult(
        fixated=fixated,
        t_fix=float(t_fix),
        t=t,
        H=H,
        p=p
    )

