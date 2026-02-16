import numpy as np

def fit_alpha(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    eps = data["eps_grid"]
    tfix = data["mean_tfix"]

    mask = (eps > 0) & np.isfinite(tfix)
    x = np.log(eps[mask])
    y = np.log(tfix[mask])

    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope

    return alpha

for fname in ["spatial_L50.npz", "spatial_sweep.npz", "spatial_L150.npz"]:
    alpha = fit_alpha(fname)
    print(f"{fname}: alpha = {alpha:.4f}")

