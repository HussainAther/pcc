#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from metrics.pcc_payoff import pcc_payoff, LABELS
from ternary_plot import plot_ternary

def replicator_rhs(x, A):

    Ax = A @ x
    phi = x @ Ax

    return x * (Ax - phi)


def simulate(A, T=20.0, dt=1e-3):

    n = A.shape[0]

    x = np.ones(n) / n
    x = x + 0.02 * np.random.default_rng(0).normal(size=n)
    x = np.clip(x, 1e-9, None)
    x = x / x.sum()

    steps = int(T / dt)

    xs = np.zeros((steps, n))
    ts = np.zeros(steps)

    for i in range(steps):

        xs[i] = x
        ts[i] = i * dt

        x = x + dt * replicator_rhs(x, A)
        x = np.clip(x, 1e-9, None)
        x = x / x.sum()

    return ts, xs


def main():

    A = pcc_payoff(mu=1.0, sigma0=1.0)

    ts, xs = simulate(A)

    plt.figure()

    for i in range(3):
        plt.plot(ts, xs[:, i], label=LABELS[i])

    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.title("PCC replicator dynamics")
    plt.legend()

    plt.savefig("pcc_replicator_timeseries.png", dpi=200)
    print("Saved pcc_replicator_timeseries.png")
    plot_ternary(xs, labels=LABELS, filename="pcc_ternary.png")


if __name__ == "__main__":
    main()

