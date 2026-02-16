import numpy as np

files = [
    "spatial_L50.npz",
    "spatial_L150.npz",
]

print("\nFinite-size scaling check\n")

for f in files:

    data = np.load(f, allow_pickle=True)

    eps = data["eps_grid"]
    tfix = data["mean_tfix"]

    print(f"\n{f}")
    print("eps:", eps)
    print("mean_tfix:", tfix)

    # filter valid entries
    mask = np.isfinite(tfix) & (eps > 0)

    x = eps[mask]
    y = tfix[mask]

    if len(x) < 2:

        print("Not enough points for exponent fit (need ≥ 2)")
        continue

    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)

    alpha = -slope

    print(f"alpha ≈ {alpha:.4f}")

