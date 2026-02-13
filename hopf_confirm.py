import numpy as np
import matplotlib.pyplot as plt
from simulation import Params, simulate

def amp(traj, transient=0.6):
    n = traj.shape[0]
    x = traj[int(transient * n):]
    # mean peak-to-peak across coordinates
    return float(np.mean(np.ptp(x, axis=0)))

# Test 1: mu > sigma0 (Stable spiral toward barycenter)
# Test 2: mu < sigma0 (Unstable interior -> sustained oscillation / limit cycle)
tests = [
    Params(mu=0.3, sigma0=0.1, sigma1=0.0, kappa=10.0, use_entropy=False),  # Stable
    Params(mu=0.1, sigma0=0.4, sigma1=0.0, kappa=10.0, use_entropy=False),  # Unstable -> LC
]

for p in tests:
    t, y = simulate(params=p, t_span=(0, 600), n_points=10000)
    A = amp(y)
    print(f"Testing sigma0={p.sigma0}, mu={p.mu}, sigma1={p.sigma1} -> Amplitude: {A:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(t, y[:, 0], label="x1")
    plt.plot(t, y[:, 1], label="x2")
    plt.plot(t, y[:, 2], label="x3")
    plt.title(f"Dynamics: σ0={p.sigma0}, μ={p.mu} (Amp={A:.4f})")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True, alpha=0.3)


