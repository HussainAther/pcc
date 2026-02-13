import numpy as np
import matplotlib.pyplot as plt

# Load data once
data = np.load("spatial_sweep.npz", allow_pickle=True)
eps = data["eps_grid"]
mean_tfix = data["mean_tfix"]
taus_by_eps = data["taus_by_eps"].item()

def fit_power_law(eps, y):
    """
    Fit y = C * eps^{-alpha} via linear regression in log space.
    Returns alpha, C, alpha_se, r2.
    """
    eps = np.asarray(eps, float)
    y = np.asarray(y, float)

    m = np.isfinite(eps) & np.isfinite(y) & (eps > 0) & (y > 0)
    x = np.log(eps[m])
    z = np.log(y[m])

    # z = a + b x, where b = -alpha
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    a, b = beta
    alpha = -b
    C = np.exp(a)

    # standard error of slope
    z_hat = X @ beta
    resid = z - z_hat
    dof = len(z) - 2
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    b_se = np.sqrt(cov[1, 1])
    alpha_se = b_se

    # R^2
    ss_tot = np.sum((z - z.mean())**2)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return alpha, C, alpha_se, r2

# --- 1. Scaling Plot ---
alpha, C, alpha_se, r2 = fit_power_law(eps, mean_tfix)
print(f"Fit: mean_tfix ≈ {C:.2e} * eps^(-{alpha:.3f})  (alpha SE={alpha_se:.3f}, R^2={r2:.4f})")

plt.figure()
plt.loglog(eps, mean_tfix, marker="o", linestyle="none")

xline = np.linspace(eps[eps>0].min(), eps.max(), 200)
yline = C * xline**(-alpha)
plt.loglog(xline, yline, linestyle="-")

plt.xlabel(r"$\epsilon$")
plt.ylabel(r"Mean fixation time $\mathbb{E}[T_{\mathrm{fix}}]$ (MCS)")
plt.title(rf"Scaling: $\mathbb{{E}}[T_{{fix}}]\propto \epsilon^{{-{alpha:.2f}}}$, $R^2={r2:.3f}$")
plt.tight_layout()
plt.savefig("spatial_sweep_scaling.png", dpi=300)
plt.close()

# --- 2. Pooled Lead-Time Distribution ---
taus_all = np.concatenate([np.asarray(v, float) for v in taus_by_eps.values() if len(v) > 0])

plt.figure()
plt.hist(taus_all, bins=30)
plt.xlabel(r"Lead time $\tau = T_{\mathrm{fix}} - T_{\mathrm{warn}}$ (MCS)")
plt.ylabel("Count")
plt.title(rf"Pooled lead-time distribution (N={len(taus_all)})")
plt.tight_layout()
plt.savefig("pooled_lead_time_dist.png", dpi=300)
plt.close()

print("Mean tau:", taus_all.mean())
print("Median tau:", np.median(taus_all))

# --- 3. Target Epsilon Distribution ---
def save_tau_hist_for_eps(eps_target=0.2, bins=25):
    keys = np.array([float(k) for k in taus_by_eps.keys()])
    k_val = float(keys[np.argmin(np.abs(keys - eps_target))])
    taus = np.asarray(taus_by_eps[str(k_val)], float)

    plt.figure()
    plt.hist(taus, bins=bins)
    plt.xlabel(r"Lead time $\tau$ (MCS)")
    plt.ylabel("Count")
    plt.title(rf"Lead-time distribution at $\varepsilon={k_val:.3f}$ (N={len(taus)})")
    plt.tight_layout()
    plt.savefig(f"lead_time_dist_eps_{k_val:.2f}.png", dpi=300)
    plt.close()

save_tau_hist_for_eps(eps_target=0.2)
