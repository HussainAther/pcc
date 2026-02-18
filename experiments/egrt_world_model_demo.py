#!/usr/bin/env python3
"""
EGRT world-model demo using PCC spatial dynamics.

Goal:
Show that model complexity required to predict system increases near instability.
Also compute a non-transitivity metric from empirical win-rates.

Outputs:
- egrt_world_model_error.png
- nontransitivity_vs_epsilon.png
"""

import numpy as np
import matplotlib.pyplot as plt

# Adjust this import to match your repo layout:
# If spatial_adapter.py is in repo root:
from spatial_adapter import run_spatial_trial
# If it really is under scripts/, use:
# from scripts.spatial_adapter import run_spatial_trial


def collect_data(
    epsilon: float,
    *,
    trials: int = 12,
    L: int = 100,
    T_max: int = 30000,
    sample_every: int = 20,
    m_sites: int = 1,
    sigma_noise: float = 0.0,
):
    """
    Runs spatial trials and returns:
      X: p(t)   stacked
      Y: p(t+1) stacked
      win_counts: empirical replacement counts (3x3)
    """
    X_list, Y_list = [], []
    win_counts = np.zeros((3, 3), dtype=np.int64)

    for seed in range(trials):
        result = run_spatial_trial(
            epsilon=epsilon,
            sigma_noise=sigma_noise,
            seed=seed,
            L=L,
            T_max=T_max,
            sample_every=sample_every,
            m_sites=m_sites,
            return_win_counts=True,
        )

        p = result.p
        if len(p) < 2:
            continue

        X_list.append(p[:-1])
        Y_list.append(p[1:])
        win_counts += result.win_counts

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y, win_counts


def fit_linear_model(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # least squares: Y ≈ X W
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def prediction_error(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float:
    Y_pred = X @ W
    return float(np.mean((Y - Y_pred) ** 2))


def dominance_from_winrate(W: np.ndarray, *, threshold: float = 0.5, margin: float = 0.02) -> np.ndarray:
    """
    W[i,j] ~ P(i replaces j).
    D[i,j]=True if i dominates j.

    We use a robust condition: W[i,j] - W[j,i] > margin.
    """
    W = np.asarray(W, float)
    D = (W - W.T) > margin
    np.fill_diagonal(D, False)
    return D


def list_3cycles(D: np.ndarray):
    """
    Return oriented triples (i,j,k) such that i->j, j->k, k->i.
    """
    D = np.asarray(D, bool)
    n = D.shape[0]
    cycles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if D[i, j] and D[j, k] and D[k, i]:
                    cycles.append((i, j, k))
                elif D[i, k] and D[k, j] and D[j, i]:
                    cycles.append((i, k, j))
    return cycles


def intransitivity_index(D: np.ndarray) -> float:
    """
    Fraction of unordered triples that form a directed 3-cycle.
    For n=3: either 1.0 (clean cycle) or 0.0 (not).
    """
    D = np.asarray(D, bool)
    n = D.shape[0]
    triples = n * (n - 1) * (n - 2) // 6
    if triples == 0:
        return 0.0
    return len(list_3cycles(D)) / triples


def main():
    # "Slow" regime so non-transitivity has time to manifest
    
    eps_values = [0.07, 0.085, 0.10, 0.12, 0.15, 0.19, 0.24, 0.30, 0.38]
    L = 100
    T_max = 20000
    sample_every = 5
    trials = 40
    m_sites = 5
    eps_gain = 15.0

    errors = []
    nontrans_scores = []

    for eps in eps_values:
        print(f"Running eps={eps:.3f}")

        X, Y, win_counts = collect_data(
            eps,
            trials=trials,
            L=L,
            T_max=T_max,
            sample_every=sample_every,
            m_sites=m_sites,
        )

        W_model = fit_linear_model(X, Y)
        errors.append(prediction_error(X, Y, W_model))

        # empirical win-rate matrix
        denom = np.maximum(win_counts + win_counts.T, 1)
        W_winrate = win_counts / denom
        np.fill_diagonal(W_winrate, 0.0)

        D = dominance_from_winrate(W_winrate, margin=0.02)
        nontrans_scores.append(intransitivity_index(D))

    errors = np.asarray(errors, float)
    nontrans_scores = np.asarray(nontrans_scores, float)

    # Plot 1: EGRT prediction error
    plt.figure()
    plt.plot(eps_values, errors, "o-", linewidth=2)
    plt.xlabel("epsilon (dissipation)")
    plt.ylabel("model prediction error")
    plt.title("EGRT: world model error vs instability")
    plt.tight_layout()
    plt.savefig("egrt_world_model_error.png", dpi=200)
    print("Saved: egrt_world_model_error.png")

    # Plot 2: Non-transitivity
    plt.figure()
    plt.plot(eps_values, nontrans_scores, "o-", linewidth=2)
    plt.xlabel("epsilon (dissipation)")
    plt.ylabel("non-transitivity index (3-cycle fraction)")
    plt.title("Non-transitivity vs epsilon (empirical win-rates)")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("nontransitivity_vs_epsilon.png", dpi=200)
    print("Saved: nontransitivity_vs_epsilon.png")


if __name__ == "__main__":
    main()

