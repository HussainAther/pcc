#!/usr/bin/env python3
"""
EGRT world-model demo using PCC spatial dynamics.

Goal:
Show that model complexity required to predict system increases near instability.
This operationalizes the Every Good Regulator Theorem.
"""

import numpy as np
import matplotlib.pyplot as plt

# Assuming this script exists in your local environment
from scripts.spatial_adapter import run_spatial_trial

# ------------------------------------------------------------
# collect trajectory data
# ------------------------------------------------------------

def collect_data(epsilon, trials=10, L=50, T_max=5000, sample_every=10):
    """
    Runs spatial trials and returns transition data (X, Y) 
    along with empirical win counts for transitivity analysis.
    """
    X = []
    Y = []

    # win_counts[i,j] = number of times i replaced j across all trials
    win_counts = np.zeros((3, 3), dtype=np.int64)

    for seed in range(trials):
        result = run_spatial_trial(
            epsilon=epsilon,
            sigma_noise=0.0,
            seed=seed,
            L=L,
            T_max=T_max,
            sample_every=sample_every,
            m_sites=1,
            return_win_counts=True,   
        )

        p = result.p
        X.append(p[:-1])
        Y.append(p[1:])

        win_counts += result.win_counts

    X = np.vstack(X)
    Y = np.vstack(Y)

    return X, Y, win_counts


# ------------------------------------------------------------
# fit linear world model
# ------------------------------------------------------------

def fit_linear_model(X, Y):
    # solve least squares: Y ≈ X W
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def prediction_error(X, Y, W):
    Y_pred = X @ W
    mse = np.mean((Y - Y_pred) ** 2)
    return mse


def dominance_from_winrate(W: np.ndarray, threshold: float = 0.5, margin: float = 0.0) -> np.ndarray:
    W = np.asarray(W, float)
    if margin > 0.0:
        D = (W - W.T) > margin
    else:
        D = W > threshold
    np.fill_diagonal(D, False)
    return D


def list_3cycles(D: np.ndarray):
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
    D = np.asarray(D, bool)
    n = D.shape[0]
    triples = n * (n - 1) * (n - 2) // 6
    if triples == 0:
        return 0.0
    return len(list_3cycles(D)) / triples


# ------------------------------------------------------------
# main experiment
# ------------------------------------------------------------

def main():
    eps_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    L = 100
    T_max = 30000
    sample_every = 20
    trials = 12
    errors = []
    nontrans_scores = [] # Initialized the missing list

    for eps in eps_values:
        print(f"Running eps={eps:.3f}")

        # Using the updated collect_data that returns 3 values
        X, Y, win_counts = collect_data(eps)

        W_model = fit_linear_model(X, Y)
        err = prediction_error(X, Y, W_model)
        errors.append(err)

        # Empirical win-rate matrix for this epsilon
        # Added safety for division by zero
        W_winrate = win_counts / np.maximum(win_counts + win_counts.T, 1)
        np.fill_diagonal(W_winrate, 0.0)

        D = dominance_from_winrate(W_winrate, threshold=0.5, margin=0.02)
        nti = intransitivity_index(D)
        nontrans_scores.append(nti) # Fixed indentation

    errors = np.array(errors)
    nontrans_scores = np.array(nontrans_scores)

    # Plot 1: Prediction Error
    plt.figure()
    plt.plot(eps_values, errors, 'o-', linewidth=2)
    plt.xlabel("epsilon (dissipation)")
    plt.ylabel("model prediction error")
    plt.title("EGRT: world model error vs instability")
    plt.savefig("egrt_world_model_error.png", dpi=200)
    print("\nSaved: egrt_world_model_error.png")

    # Plot 2: Non-transitivity
    plt.figure()
    plt.plot(eps_values, nontrans_scores, "o-", linewidth=2)
    plt.xlabel("epsilon (dissipation)")
    plt.ylabel("non-transitivity index (3-cycle fraction)")
    plt.title("Non-transitivity vs epsilon (empirical win-rates)")
    plt.ylim(-0.05, 1.05)
    plt.savefig("nontransitivity_vs_epsilon.png", dpi=200)
    print("Saved: nontransitivity_vs_epsilon.png")

if __name__ == "__main__":
    main()
