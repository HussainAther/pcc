import argparse
import numpy as np
import matplotlib.pyplot as plt


def fit_power_law(eps, y):
    eps = np.asarray(eps, float)
    y = np.asarray(y, float)

    mask = np.isfinite(eps) & np.isfinite(y) & (eps > 0) & (y > 0)
    eps = eps[mask]
    y = y[mask]

    x = np.log(eps)
    z = np.log(y)

    slope, intercept = np.polyfit(x, z, 1)
    alpha = -slope
    C = np.exp(intercept)

    zhat = slope * x + intercept
    ss_res = np.sum((z - zhat) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # rough SE for alpha
    n = len(x)
    if n > 2:
        s2 = ss_res / (n - 2)
        Sxx = np.sum((x - np.mean(x)) ** 2)
        alpha_se = np.sqrt(s2 / Sxx)
    else:
        alpha_se = np.nan

    return alpha, C, alpha_se, r2


def closest_key(d, target):
    # keys might be floats; choose the closest one
    keys = np.array(list(d.keys()), float)
    if len(keys) == 0:
        return None
    return float(keys[np.argmin(np.abs(keys - target))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="spatial_sweep_clean2.npz",
                    help="NPZ file from run_spatial_sweep.py")
    ap.add_argument("--eps_for_tau", type=float, default=0.2,
                    help="epsilon value to show tau distribution near")
    # Added the --out argument here
    ap.add_argument("--out", default="spatial_sweep_scaling.png",
                    help="filename for the scaling plot")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    eps = data["eps_grid"]
    mean_tfix = data["mean_tfix"]
    taus_by_eps = data.get("taus_by_eps", None)
    if taus_by_eps is not None:
        taus_by_eps = taus_by_eps.item()

    alpha, C, alpha_se, r2 = fit_power_law(eps, mean_tfix)
    print(f"Fit: mean_tfix ≈ {C:.2e} * eps^(-{alpha:.3f})  (alpha SE={alpha_se:.3f}, R^2={r2:.4f})")

    # plot scaling
    mask = np.isfinite(mean_tfix) & (eps > 0)
    x = eps[mask]
    y = mean_tfix[mask]

    plt.figure()
    plt.scatter(x, y)

    # fitted line
    xs = np.linspace(x.min(), x.max(), 200)
    ys = C * xs ** (-alpha)
    plt.plot(xs, ys)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"Mean collapse time $E[T]$ (MCS)")
    plt.title(rf"Scaling: $E[T]\propto \epsilon^{{-{alpha:.2f}}}$, $R^2={r2:.3f}$")
    plt.tight_layout()
    
    # Use the argument here instead of the hardcoded string
    plt.savefig(args.out, dpi=200)
    
    # pooled tau distribution (safe even if empty)
    if taus_by_eps is None:
        print("No lead-time dictionary found in NPZ; skipping tau plots.")
        return

    tau_arrays = [np.asarray(v, float) for v in taus_by_eps.values() if len(v) > 0]
    if len(tau_arrays) == 0:
        print("No lead-time data found; skipping pooled distribution plot.")
        return

    taus_all = np.concatenate(tau_arrays)

    plt.figure()
    plt.hist(taus_all, bins=30)
    plt.xlabel(r"Lead time $\tau$ (MCS)")
    plt.ylabel("Count")
    plt.title(f"Pooled lead-time distribution (N={len(taus_all)})")
    plt.tight_layout()
    plt.savefig("pooled_lead_time_dist.png", dpi=200)

    # show tau distribution for epsilon ~ eps_for_tau (closest key)
    k = closest_key(taus_by_eps, args.eps_for_tau)
    if k is None or len(taus_by_eps.get(k, [])) == 0:
        print(f"No tau data near epsilon={args.eps_for_tau}.")
        return

    taus_k = np.asarray(taus_by_eps[k], float)
    plt.figure()
    plt.hist(taus_k, bins=25)
    plt.xlabel(r"Lead time $\tau$ (MCS)")
    plt.ylabel("Count")
    plt.title(f"Lead-time distribution near ε={k:.3f} (N={len(taus_k)})")
    plt.tight_layout()
    plt.savefig(f"lead_time_dist_eps_{k:.3f}.png", dpi=200)


if __name__ == "__main__":
    main()
