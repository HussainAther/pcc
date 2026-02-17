#!/usr/bin/env python3
import numpy as np

LABELS = ["Pressure", "Chaos", "Control"]

def pcc_payoff(mu: float = 1.0, sigma0: float = 1.0) -> np.ndarray:
    A = np.array([
        [0.0, +mu, -mu],
        [-mu, 0.0, +mu],
        [+mu, -mu, 0.0],
    ], dtype=float)
    A[0, 1] *= sigma0
    return A

def dominance_from_payoff(A: np.ndarray, margin: float = 0.0) -> np.ndarray:
    diff = A - A.T
    D = diff > margin
    np.fill_diagonal(D, False)
    return D

def list_3cycle(D: np.ndarray):
    # For 3x3, just check the two possible cycles
    if D[0,1] and D[1,2] and D[2,0]:
        return (0,1,2)
    if D[0,2] and D[2,1] and D[1,0]:
        return (0,2,1)
    return None

def main():
    A = pcc_payoff(mu=1.0, sigma0=1.0)
    D = dominance_from_payoff(A)

    print("Dominance adjacency D (1 means row beats col):")
    print(D.astype(int))
    print()

    cyc = list_3cycle(D)
    if cyc is None:
        print("No clean 3-cycle detected (possible ties or transitive ordering).")
    else:
        i,j,k = cyc
        print(f"3-cycle: {LABELS[i]} → {LABELS[j]} → {LABELS[k]} → {LABELS[i]}")

if __name__ == "__main__":
    main()

