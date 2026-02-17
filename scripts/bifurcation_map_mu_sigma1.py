# bifurcation_map_mu_sigma1.py - REVISED

import numpy as np
import matplotlib.pyplot as plt
from simulation import Params, simulate

def amplitude(traj: np.ndarray, transient_frac: float = 0.5) -> float:
    n = traj.shape[0]
    start = int(transient_frac * n)
    x = traj[start:, :]
    # ptp (peak-to-peak) is fine, but we average across the 3 components
    return float(np.mean(np.ptp(x, axis=0)))

def robust_threshold(values, k=10.0):
    v = np.asarray(values)
    med = np.median(v)
    # MAD (Median Absolute Deviation) provides a robust noise floor estimate
    mad = np.median(np.abs(v - med)) + 1e-15
    return med + k * mad

def main():
    mus = np.linspace(0.0, 0.30, 31)
    sigma1s = np.linspace(0.0, 2.5, 41)
    
    kappa = 5.0
    sigma0 = 0.0
    A = np.zeros((len(mus), len(sigma1s)), dtype=float)

    print("Running simulations...")
    for i, mu in enumerate(mus):
        for j, s1 in enumerate(sigma1s):
            p = Params(mu=float(mu), kappa=kappa, sigma0=sigma0, sigma1=float(s1), use_entropy=False)
            # Increased n_points for better resolution of tiny oscillations
            t, y = simulate(y0=(0.33,0.33,0.34), t_span=(0, 800), n_points=16000, params=p)
            A[i, j] = amplitude(y)

    # --- THRESHOLD CALCULATION ---
    # We look at the "low sigma, high mu" corner where we expect stability
    mu_grid, s1_grid = np.meshgrid(mus, sigma1s, indexing="ij")
    stable_region_mask = (s1_grid <= 0.2) & (mu_grid >= 0.1)
    A_thr = robust_threshold(A[stable_region_mask], k=8.0)
    print(f"Calculated Noise Floor Threshold: {A_thr:.6f}")

    # --- PLOTTING ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap
    im = ax[0].imshow(
        A, origin="lower", aspect="auto",
        extent=[sigma1s[0], sigma1s[-1], mus[0], mus[-1]],
        cmap='magma'
    )
    plt.colorbar(im, ax=ax[0], label="Amplitude")
    ax[0].set_title("Bifurcation Heatmap")
    ax[0].set_xlabel("$\sigma_1$ (Endogenous Gain)")
    ax[0].set_ylabel("$\mu$ (Leakage)")

    # Thresholded Boundary
    ax[1].contourf(
        sigma1s, mus, A, 
        levels=[0, A_thr, np.max(A)], 
        colors=['#2c3e50', '#e74c3c'], 
        alpha=0.8
    )
    ax[1].set_title(f"Stable (Blue) vs Oscillatory (Red)\nThreshold: {A_thr:.2e}")
    ax[1].set_xlabel("$\sigma_1$")
    ax[1].set_ylabel("$\mu$")

    plt.tight_layout()
    plt.savefig("refined_bifurcation_map.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
