#!/usr/bin/env python3
"""
scripts/nontransitivity_vs_epsilon.py

Compute a non-transitivity metric vs epsilon for the spatial PCC/RPS lattice.

Idea:
- For each epsilon, run multiple trials
- Accumulate empirical win-counts: win_counts[i,j] = # times i replaced j
- Convert to a robust dominance relation using margin on (W - W^T)
- Compute nontransitivity index = fraction of directed 3-cycles
  For n=3 strategies, this is typically 1.0 if the dominance graph is a clean cycle,
  and 0.0 if dominance collapses into a transitive/degenerate relation (often due to rapid extinction).

Outputs:
- nontransitivity_vs_epsilon.png
- (optional) nontransitivity_vs_epsilon.npz with eps_values + scores + win_counts
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


# Robust import to match your repo layout
try:
    from scripts.spatial_adapter import run_spatial_trial
except Exception:
    from spatial_adapter import run_spatial_trial


def dominance_from_winrate(W: np.ndarray, *, margin: float = 0.02) -> np.ndarray:
    """
    Build dominance adjacency D from empirical win-rates W.
    D[i,j]=True means i dominates j.

    We use a robust rule:
        i beats j if W[i,j] - W[j,i] > margin
    """
    W = np.asarray(W, float)
    D = (W - W.T) > float(margin)
    np.fill_diagonal(D, False)
    return D


def list_3cycles(D: np.ndarray):
    """
    Enumerate directed 3-cycles (i->j, j->k, k->i) on unordered triples.
    Returns oriented triples.
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
    For n=3 strategies: either 1.0 (clean cycle) or 0.0 (not clean / too many ties).
    """
    D = np.asarray(D, bool)
    n = D.shape[0]
    triples = n * (n - 1) * (n - 2) // 6
    if triples == 0:
        return 0.0
    return len(list_3cycles(D)) / triples


def parse_eps_list(eps_args):
    # Allow either: --eps_values 0.07 0.10 ...
    # or: --eps_values "0.07,0.10,0.12"
    if len(eps_args) == 1 and ("," in eps_args[0]):
        return [float(x.strip()) for x in eps_args[0].split(",") if x.strip()]
    return [float(x) for x in eps_args]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--T_max", type=int, default=30000)
    ap.add_argument("--sample_every", type=int, default=20)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--m_sites", type=int, default=1)
    ap.add_argument("--sigma_noise", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument(
        "--eps_values",
        nargs="+",
        default=["0.005,0.01,0.015,0.02,0.03,0.04,0.05"],
        help="List of eps values (space-separated) or a single comma-separated string.",
    )
    ap.add_argument("--out_png", type=str, default="nontransitivity_vs_epsilon.png")
    ap.add_argument("--out_npz", type=str, default="nontransitivity_vs_epsilon.npz")
    ap.add_argument("--no_save_npz", action="store_true", help="Do not save the .npz output")
    args = ap.parse_args()

    eps_values = parse_eps_list(args.eps_values)

    scores = []
    win_counts_all = []

    print("===================================")
    print("Non-transitivity sweep starting")
    print(f"L={args.L}, trials={args.trials}, T_max={args.T_max}, m_sites={args.m_sites}")
    print(f"eps_values={eps_values}")
    print("===================================")

    for eps in eps_values:
        win_counts = np.zeros((3, 3), dtype=np.int64)

        for seed in range(args.trials):
            r = run_spatial_trial(
                epsilon=float(eps),
                sigma_noise=float(args.sigma_noise),
                seed=int(seed),
                L=int(args.L),
                T_max=int(args.T_max),
                sample_every=int(args.sample_every),
                m_sites=int(args.m_sites),
                return_win_counts=True,
            )
            win_counts += np.asarray(r.win_counts, dtype=np.int64)

        denom = np.maximum(win_counts + win_counts.T, 1)
        W = win_counts / denom
        np.fill_diagonal(W, 0.0)

        D = dominance_from_winrate(W, margin=float(args.margin))
        nti = intransitivity_index(D)

        scores.append(float(nti))
        win_counts_all.append(win_counts)

        print(f"eps={eps:.3f} | nontrans_index={nti:.3f} | win_counts=\n{win_counts}")

    scores = np.array(scores, dtype=float)

    # Plot
    plt.figure()
    plt.plot(eps_values, scores, "o-", linewidth=2)
    plt.xlabel("epsilon (dissipation)")
    plt.ylabel("non-transitivity index (3-cycle fraction)")
    plt.title("Non-transitivity vs epsilon (empirical win-rates)")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"\nSaved: {args.out_png}")

    # Save data
    if not args.no_save_npz:
        np.savez(
            args.out_npz,
            eps_values=np.array(eps_values, float),
            nontransitivity=scores,
            win_counts_by_eps=np.array(win_counts_all, dtype=np.int64),
            L=args.L,
            T_max=args.T_max,
            sample_every=args.sample_every,
            trials=args.trials,
            m_sites=args.m_sites,
            sigma_noise=args.sigma_noise,
            margin=args.margin,
        )
        print(f"Saved: {args.out_npz}")


if __name__ == "__main__":
    main()

