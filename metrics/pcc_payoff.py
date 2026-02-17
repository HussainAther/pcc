"""
pcc_payoff.py

Defines PCC payoff matrices and dominance extraction.

Strategy order:
0 = Pressure
1 = Chaos
2 = Control

Cycle:
Pressure → Chaos → Control → Pressure
"""

import numpy as np

LABELS = ["Pressure", "Chaos", "Control"]


def pcc_payoff(mu: float = 1.0, sigma0: float = 1.0) -> np.ndarray:
    """
    PCC payoff matrix.

    A[i,j] > 0 means strategy i beats j.
    sigma0 introduces asymmetry (dissipation parameter).
    """

    A = np.array([
        [0.0, +mu, -mu],  # Pressure beats Chaos
        [-mu, 0.0, +mu],  # Chaos beats Control
        [+mu, -mu, 0.0],  # Control beats Pressure
    ], dtype=float)

    # asymmetry dial (dissipation)
    A[0, 1] *= sigma0

    return A


def dominance_matrix(A: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Convert payoff matrix into dominance adjacency.
    """
    diff = A - A.T
    D = diff > margin
    np.fill_diagonal(D, False)
    return D


def print_interactions(A: np.ndarray):
    """
    Human-readable interaction summary.
    """

    print("PCC interactions:\n")

    for i in range(3):
        for j in range(3):
            if i != j and A[i, j] > 0:
                print(f"{LABELS[i]} beats {LABELS[j]} (A[{i},{j}]={A[i,j]:+.3f})")

    print("\nCycle: Pressure → Chaos → Control → Pressure\n")

