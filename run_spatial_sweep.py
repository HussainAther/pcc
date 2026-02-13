# run_spatial_sweep.py
from __future__ import annotations
import argparse
import numpy as np
from spatial_adapter import run_spatial_trial
from lead_time import warn_time_from_trace

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps_min", type=float, default=0.0)
    ap.add_argument("--eps_max", type=float, default=0.4)
    ap.add_argument("--eps_n", type=int, default=9)
    ap.add_argument("--sigma", type=float, default=0.10, help="noise level")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--T_max", type=int, default=50_000)
    ap.add_argument("--sample_every", type=int, default=50)
    ap.add_argument("--out", type=str, default="spatial_sweep.npz")
    args = ap.parse_args()

    eps_grid = np.linspace(args.eps_min, args.eps_max, args.eps_n)

    fix_prob = np.zeros_like(eps_grid)
    mean_tfix = np.zeros_like(eps_grid)
    median_tfix = np.zeros_like(eps_grid)

    example_traces = {}
    taus_by_eps = {}
    thr_by_eps = {}

    for i, eps in enumerate(eps_grid):
        fixes = []
        tfixes = []
        traces = []
        taus = []
        used_thrs = []

        for k in range(args.trials):
            seed = args.seed + 10_000 * i + k
            res = run_spatial_trial(
                epsilon=float(eps),
                sigma_noise=float(args.sigma),
                seed=int(seed),
                L=args.L,
                T_max=args.T_max,
                sample_every=args.sample_every,
            )

            fixes.append(res.fixated)
            tfixes.append(res.t_fix)

            # Syntax fixed: t_stop is now defined in lead_time.py
            Twarn, used_thr = warn_time_from_trace(
                res.t, res.H,
                k=5,
                threshold=None,
                threshold_quantile=0.20,
                smooth_w=9,
                t_stop=res.t_fix if res.fixated else None
            )
            used_thrs.append(used_thr)

            if res.fixated and (Twarn is not None) and (res.t_fix >= Twarn):
                taus.append(float(res.t_fix - Twarn))

            if len(traces) < 5:
                traces.append({
                    "t": res.t, "H": res.H, "p": res.p,
                    "fixated": res.fixated, "t_fix": res.t_fix,
                    "Twarn": Twarn, "used_thr": used_thr
                })

        fixes_arr = np.array(fixes, dtype=bool)
        tfixes_arr = np.array(tfixes, dtype=float)

        fix_prob[i] = fixes_arr.mean()
        if fixes_arr.any():
            mean_tfix[i] = tfixes_arr[fixes_arr].mean()
            median_tfix[i] = np.median(tfixes_arr[fixes_arr])
        else:
            mean_tfix[i] = np.nan
            median_tfix[i] = np.nan

        example_traces[str(eps)] = traces
        taus_by_eps[str(eps)] = np.array(taus, dtype=float)
        thr_by_eps[str(eps)] = np.array(used_thrs, dtype=float)

        print(f"eps={eps:.3f} | fix_prob={fix_prob[i]:.2f} | mean_tfix={mean_tfix[i]:.1f} | n_taus={len(taus)}")

    np.savez_compressed(
        args.out,
        eps_grid=eps_grid,
        sigma=args.sigma,
        trials=args.trials,
        fix_prob=fix_prob,
        mean_tfix=mean_tfix,
        median_tfix=median_tfix,
        example_traces=example_traces,
        taus_by_eps=taus_by_eps,
        thr_by_eps=thr_by_eps,
    )
    print(f"Results saved to {args.out}")

if __name__ == "__main__":
    main()
