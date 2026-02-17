#!/usr/bin/env python3
import numpy as np

LABELS = ["Pressure", "Chaos", "Control"]

def pcc_payoff(mu: float = 1.0, sigma0: float = 1.0) -> np.ndarray:
    A = np.array([
        [0.0, +mu, -mu],  # P beats Ch, loses to Co
        [-mu, 0.0, +mu],  # Ch beats Co, loses to P
        [+mu, -mu, 0.0],  # Co beats P, loses to Ch
    ], dtype=float)
    A[0, 1] *= sigma0  # bias P->Ch edge if sigma0>1
    return A

def main():
    A = pcc_payoff(mu=1.0, sigma0=1.0)
    print("Payoff matrix A (A[i,j] > 0 means i beats j):")
    print(A)
    print()

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if A[i, j] > 0:
                print(f"{LABELS[i]} beats {LABELS[j]} (A[{i},{j}]={A[i,j]:+.2f})")

    print("\nCycle summary:")
    print("Pressure → Chaos → Control → Pressure")

if __name__ == "__main__":
    main()

