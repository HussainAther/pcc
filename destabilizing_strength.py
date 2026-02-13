import numpy as np
import matplotlib.pyplot as plt
from simulation import Params, simulate

def calculate_amplitude(params, t_span=(0, 600)):
    """Runs a simulation and returns the peak-to-peak amplitude of the limit cycle."""
    t, y = simulate(params=params, t_span=t_span)
    # Use the last 20% of the trajectory to ensure transients have died out
    cutoff = int(len(y) * 0.8)
    tail = y[cutoff:, :]
    
    # Calculate peak-to-peak for each strategy and average them
    ptp = np.ptp(tail, axis=0)
    return np.mean(ptp)

def main():
    # Set mu to a fixed value; the bifurcation should occur at sigma = mu
    fixed_mu = 0.1
    kappa_val = 10.0
    
    # Range of sigma values crossing the bifurcation point (0.1)
    sigmas = np.linspace(0.0, 0.5, 30)
    amplitudes = []

    print(f"Starting sweep (mu={fixed_mu})...")
    for s in sigmas:
        p = Params(mu=fixed_mu, sigma0=float(s), sigma1=0.0, kappa=kappa_val, use_entropy=False)
        a = calculate_amplitude(p)
        amplitudes.append(a)
        print(f"sigma: {s:.3f} | amplitude: {a:.6f}")

    # Plotting the results
    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, amplitudes, 'o-', linewidth=2, markersize=5, label="Simulated Amplitude")
    
    # Theoretical threshold line
    plt.axvline(fixed_mu, color='r', linestyle='--', label=f"Bifurcation Threshold (σ={fixed_mu})")
    
    plt.title("Hopf Bifurcation: Amplitude vs. Destabilizing Strength (σ)")
    plt.xlabel("Destabilizing Strength (σ)")
    plt.ylabel("Mean Steady-State Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save for the paper
    plt.savefig("hopf_amplitude_growth.png", dpi=300)

if __name__ == "__main__":
    main()
