#!/usr/bin/env python3
"""
run_spatial_sweep.py

Spatial stochastic sweep over epsilon values.

Computes:
    - fixation probability
    - mean fixation time
    - entropy early-warning lead times (tau)

Uses spatial_adapter.run_spatial_trial()

Outputs .npz file containing:
    eps_grid
    mean_tfix
    fix_prob
    taus_by_eps
    metadata (L, T_max, sample_every, trials)
"""

import argparse
import numpy as np

from spatial_adapter import run_spatial_trial


# ============================================================
# Argument parsing
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    # epsilon grid options
    parser.add_argument("--eps_min", type=float, default=0.0)
    parser.add_argument("--eps_max", type=float, default=0.4)
    parser.add_argument("--eps_n", type=int, default=9)

    parser.add_argument(
        "--eps_values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit epsilon values (overrides eps_min/max/n)"
    )

    # simulation parameters
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m_sites", type=int, default=5) # Kept this, removed the duplicates above it

    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--T_max", type=int, default=50000)
    parser.add_argument("--sample_every", type=int, default=50)

    parser.add_argument("--out", type=str, default="spatial_sweep.npz")

    return parser.parse_args()


# ============================================================
# Lead time detection
# ============================================================

def compute_tau(t, H, t_fix, threshold_quantile=0.2, k=5):
    """
    Compute early-warning lead time tau = T_fix − T_warn

    Warning defined as sustained entropy-rate decline.
    """

    t = np.asarray(t, float)
    H = np.asarray(H, float)

    if len(t) < k + 2:
        return None

    dH = np.gradient(H, t)

    finite = np.isfinite(dH)
    if finite.sum() < k:
        return None

    threshold = np.quantile(dH[finite], threshold_quantile)
    threshold = min(threshold, 0.0)

    consec = 0

    for i, val in enumerate(dH):

        if val < threshold:
            consec += 1

            if consec >= k:
                warn_idx = i - (k - 1)
                t_warn = t[warn_idx]
                return t_fix - t_warn

        else:
            consec = 0

    return None


# ============================================================
# Main sweep
# ============================================================

def main():

    args = parse_args()

    rng = np.random.default_rng(args.seed)

    # choose epsilon grid
    if args.eps_values is not None:
        eps_grid = np.array(args.eps_values, dtype=float)
    else:
        eps_grid = np.linspace(args.eps_min, args.eps_max, args.eps_n)

    mean_tfix = []
    fix_prob = []
    taus_by_eps = {}

    print("\n===================================")
    print("Spatial sweep starting")
    print(f"L={args.L}, trials={args.trials}, T_max={args.T_max}")
    print("===================================")

    for eps in eps_grid:

        print(f"\nRunning eps = {eps:.3f}")

        tfix_list = []
        taus = []

        for trial in range(args.trials):

            if (trial + 1) % 5 == 0:
                print(f"  trial {trial+1}/{args.trials}", flush=True)

            seed = int(rng.integers(0, 2**31 - 1))

            result = run_spatial_trial(
                epsilon=eps,
                sigma_noise=args.sigma,
                seed=seed,
                L=args.L,
                T_max=args.T_max,
                sample_every=args.sample_every,
                m_sites=args.m_sites
            )

            if result.fixated:

                tfix_list.append(result.t_fix)

                tau = compute_tau(
                    result.t,
                    result.H,
                    result.t_fix
                )

                if tau is not None:
                    taus.append(tau)

        # compute statistics
        if len(tfix_list) > 0:
            mean_fix = float(np.mean(tfix_list))
            prob_fix = len(tfix_list) / args.trials
        else:
            mean_fix = np.nan
            prob_fix = 0.0

        mean_tfix.append(mean_fix)
        fix_prob.append(prob_fix)

        taus_by_eps[str(eps)] = np.array(taus, dtype=float)

        print(
            f"eps={eps:.3f} | "
            f"fix_prob={prob_fix:.2f} | "
            f"mean_tfix={mean_fix:.1f} | "
            f"n_taus={len(taus)}"
        )

    # save results
    np.savez(
        args.out,
        eps_grid=np.array(eps_grid, dtype=float),
        mean_tfix=np.array(mean_tfix, dtype=float),
        fix_prob=np.array(fix_prob, dtype=float),
        taus_by_eps=taus_by_eps,
        L=args.L,
        T_max=args.T_max,
        sample_every=args.sample_every,
        trials=args.trials,
        m_sites=args.m_sites,
    )

    print("\n===================================")
    print(f"Results saved to {args.out}")
    print("===================================")


# ============================================================

if __name__ == "__main__":
    main()

