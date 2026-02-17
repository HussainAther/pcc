#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

LABELS = ["Pressure", "Chaos", "Control"]

def pcc_payoff(mu: float = 1.0, sigma0: float = 1.0) -> np.ndarray:
    A = np.array([
        [0.0, +mu, -mu],
        [-mu, 0.0, +mu],
        [+mu, -mu, 0.0],
    ], dtype=float)
    A[0, 1] *= sigma0
    return A

def replicator_rhs(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    Ax = A @ x
    phi = float(x @ Ax)
    return x * (Ax - phi)

def simulate(A: np.ndarray, x0=None, dt=1e-3, T=20.0, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    if x0 is None:
        x = np.ones(n) / n
        x = x + 0.02 * rng.normal(size=n)
        x = np.clip(x, 1e-9, None)
        x = x / x.sum()
    else:
        x = np.asarray(x0, float)
        x = np.clip(x, 1e-9, None)
        x = x / x.sum()

    steps = int(T / dt)
    xs = np.zeros((steps + 1, n))
    ts = np.linspace(0, T, steps + 1)
    xs[0] = x

    for i in range(steps):
        x = x + dt * replicator_rhs(x, A)
        x = np.clip(x, 1e-12, None)
        x = x / x.sum()
        xs[i+1] = x

    return ts, xs

def main():
    A = pcc_payoff(mu=1.0, sigma0=1.0)  # try sigma0=1.2 for biased PCC
    ts, xs = simulate(A, T=20.0, dt=1e-3)

    plt.figure()
    for i in range(3):
        plt.plot(ts, xs[:, i], label=LABELS[i])
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.title("Replicator dynamics (PCC / cyclic dominance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("replicator_pcc_timeseries.png", dpi=200)
    print("Saved: replicator_pcc_timeseries.png")

if __name__ == "__main__":
    main()

