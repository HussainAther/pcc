# stress_test.py
import numpy as np
import matplotlib.pyplot as plt
from simulation import Params, simulate, max_realpart_eigs_at_eq

def fixation_occurred(traj: np.ndarray, eps: float = 1e-3) -> bool:
    # traj shape: (T, 3)
    # Checks if any population component drops below the epsilon threshold
    return np.any(np.min(traj, axis=1) < eps)

def oscillation_amplitude(traj: np.ndarray, transient_frac: float = 0.3) -> float:
    n = traj.shape[0]
    start = int(transient_frac * n)
    x = traj[start:, :]
    # peak-to-peak averaged across components to measure limit cycle breadth
    amp = np.mean(np.ptp(x, axis=0))
    return float(amp)

def main():
    # Sweep mu to look for a Hopf-like transition or stability shifts
    mus = np.linspace(0.0, 0.2, 40)
    amps = []
    fix = []
    max_re = []

    # Updated to match current Params signature
    # Assuming default sigma values and entropy disabled for the stress test
    base = Params(mu=0.0, kappa=5.0, sigma0=0.0, sigma1=0.0, use_entropy=False)

    rng = np.random.default_rng(0)
    n_trials = 10

    for mu in mus:
        p = Params(
            mu=float(mu), 
            kappa=base.kappa, 
            sigma0=base.sigma0, 
            sigma1=base.sigma1, 
            use_entropy=base.use_entropy
        )

        amp_trials = []
        fix_trials = []
        for _ in range(n_trials):
            # slight random perturbation of initial condition to test robustness
            y0 = np.array([0.33, 0.33, 0.34]) + rng.normal(0, 0.002, size=3)
            y0 = np.clip(y0, 1e-6, 1.0)
            y0 = y0 / y0.sum()

            t, y = simulate(y0=tuple(y0), t_span=(0, 300), n_points=6000, params=p)

            amp_trials.append(oscillation_amplitude(y))
            fix_trials.append(1.0 if fixation_occurred(y, eps=1e-4) else 0.0)

        amps.append(float(np.mean(amp_trials)))
        fix.append(float(np.mean(fix_trials)))
        
        # Using the existing stability function from simulation.py
        max_re.append(max_realpart_eigs_at_eq(p))

    # Amplitude Plot
    plt.figure(figsize=(9,4))
    plt.plot(mus, amps, "o-", color="royalblue", label="Observed Amplitude")
    plt.xlabel("mixing/leakage μ")
    plt.ylabel("mean oscillation amplitude")
    plt.title("Oscillation amplitude vs μ (stochastic initial conditions)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("bifurcation_amplitude_vs_mu.png", dpi=300)
    plt.close()

    # Stability Plot
    plt.figure(figsize=(9,4))
    plt.plot(mus, max_re, "o-", color="crimson")
    plt.axhline(0.0, color="black", linestyle="--")
    plt.xlabel("mixing/leakage μ")
    plt.ylabel("max Re(λ) at interior equilibrium")
    plt.title("Local stability indicator vs μ")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("stability_max_realpart_vs_mu.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
