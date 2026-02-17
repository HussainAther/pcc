import numpy as np

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


data = np.load("spatial_sweep.npz", allow_pickle=True)
eps = data["eps_grid"]
mean_tfix = data["mean_tfix"]

alpha, C, alpha_se, r2 = fit_power_law(eps, mean_tfix)
print(f"Fit: mean_tfix ≈ {C:.2e} * eps^(-{alpha:.3f})  (alpha SE={alpha_se:.3f}, R^2={r2:.4f})")

