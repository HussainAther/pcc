import numpy as np
import matplotlib.pyplot as plt
from simulation import Params, simulate

def oscillation_power(traj: np.ndarray, transient_frac: float = 0.5) -> float:
    n = traj.shape[0]
    start = int(transient_frac * n)
    # Focus on one component and remove DC offset
    x = traj[start:, 0] 
    x = x - np.mean(x)
    
    if len(x) < 2: return 0.0
    
    # Compute Power Spectral Density via FFT
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals)**2
    # Return mean power (ignoring bin 0)
    return float(np.sum(power[1:]) / len(x))

def robust_threshold(values, k=10.0):
    v = np.asarray(values)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-18
    return med + k * mad

def main():
    # Increase resolution slightly to see the curve clearly
    mus = np.linspace(0.0, 0.40, 41)
    sigma1s = np.linspace(0.0, 5.0, 51)
    
    kappa = 5.0
    P = np.zeros((len(mus), len(sigma1s)), dtype=float)

    print("Scanning parameter space (using FFT power)...")
    for i, mu in enumerate(mus):
        for j, s1 in enumerate(sigma1s):
            params = Params(mu=float(mu), kappa=kappa, sigma0=0.0, sigma1=float(s1))
            # Long enough t_span to ensure transients die out
            t, y = simulate(y0=(0.33, 0.33, 0.34), t_span=(0, 1000), n_points=10000, params=params)
            P[i, j] = oscillation_power(y)

    # Calculate threshold from the high-leakage (stable) regime
    mu_grid, s1_grid = np.meshgrid(mus, sigma1s, indexing="ij")
    stable_mask = (mu_grid >= 0.2) & (s1_grid <= 1.0)
    p_thr = robust_threshold(P[stable_mask], k=15.0)
    
    print(f"Power threshold: {p_thr:.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Log-Power Heatmap
    im = ax[0].imshow(
        np.log10(P + 1e-18), origin="lower", aspect="auto",
        extent=[sigma1s[0], sigma1s[-1], mus[0], mus[-1]], cmap='magma'
    )
    plt.colorbar(im, ax=ax[0], label="log10(Spectral Power)")
    ax[0].set_title("Spectral Energy Map")
    ax[0].set_xlabel("$\sigma_1$")
    ax[0].set_ylabel("$\mu$")

    # Right: The Phase Diagram (The "Money Shot" for the paper)
    ax[1].contourf(
        sigma1s, mus, P, levels=[0, p_thr, np.max(P)], 
        colors=['#2c3e50', '#f1c40f'], alpha=0.9
    )
    ax[1].set_title("System Phase Diagram")
    ax[1].set_xlabel("$\sigma_1$ (Endogenous Gain)")
    ax[1].set_ylabel("$\mu$ (Leakage)")
    # Add a custom legend or text
    ax[1].text(0.5, 0.3, 'STABLE', color='white', fontweight='bold')
    ax[1].text(3.5, 0.05, 'OSCILLATORY', color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig("final_bifurcation_map.png", dpi=300)

if __name__ == "__main__":
    main()
