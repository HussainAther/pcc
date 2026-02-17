# sweep_beta_mu.py
import numpy as np
import matplotlib.pyplot as plt

from simulation import PCCParams, simulate, interior_equilibrium, jacobian_at

def oscillation_amplitude(traj: np.ndarray, transient_frac: float = 0.4) -> float:
    n = traj.shape[0]
    start = int(transient_frac * n)
    x = traj[start:, :]
    return float(np.mean(np.ptp(x, axis=0)))  # mean peak-to-peak over components

def max_realpart_eigs_at_eq(params: PCCParams) -> float:
    x_star = interior_equilibrium(params)   # your mu>0 numeric solver handles asymmetry
    J = jacobian_at(x_star, params)
    eig = np.linalg.eigvals(J)
    return float(np.max(np.real(eig)))

def run_grid(
    betas,
    mus,
    alpha=1.0,
    gamma=1.0,
    y0=(0.33, 0.33, 0.34),
    t_span=(0.0, 600.0),
    n_points=12000,
    n_trials=5,
    seed=0,
):
    rng = np.random.default_rng(seed)

    max_re = np.zeros((len(mus), len(betas)), dtype=float)
    amp = np.zeros((len(mus), len(betas)), dtype=float)

    for i, mu in enumerate(mus):
        for j, beta in enumerate(betas):
            p = PCCParams(alpha=float(alpha), beta=float(beta), gamma=float(gamma), mu=float(mu))

            # eigenvalue indicator at equilibrium
            max_re[i, j] = max_realpart_eigs_at_eq(p)

            # amplitude averaged over trials (tiny IC noise to avoid accidental symmetry)
            amps = []
            for _ in range(n_trials):
                y0n = np.array(y0, dtype=float) + rng.normal(0, 0.002, size=3)
                y0n = np.clip(y0n, 1e-6, 1.0)
                y0n = y0n / y0n.sum()
                t, y = simulate(y0=tuple(y0n), t_span=t_span, n_points=n_points, params=p)
                amps.append(oscillation_amplitude(y))

            amp[i, j] = float(np.mean(amps))

    return max_re, amp

def heatmap(data, betas, mus, title, cbar_label, fname):
    plt.figure(figsize=(10, 5))
    plt.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[betas[0], betas[-1], mus[0], mus[-1]],
    )
    plt.xlabel("Control strength β (Co > P)")
    plt.ylabel("Mixing/leakage μ")
    plt.title(title)
    plt.colorbar(label=cbar_label)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

def main():
    # Sweep ranges (start moderate; widen once you see structure)
    betas = np.linspace(0.5, 3.0, 45)     # Control dominance sweep
    mus   = np.linspace(0.0, 0.20, 41)    # leakage sweep

    max_re, amp = run_grid(
        betas=betas,
        mus=mus,
        alpha=1.0,
        gamma=1.0,
        t_span=(0.0, 600.0),
        n_points=12000,
        n_trials=5,
        seed=0,
    )

    heatmap(
        max_re,
        betas,
        mus,
        title="Stability indicator: max Re(λ) at interior equilibrium",
        cbar_label="max Re(λ)",
        fname="grid_max_realpart_beta_mu.png",
    )

    heatmap(
        amp,
        betas,
        mus,
        title="Post-transient oscillation amplitude (mean peak-to-peak)",
        cbar_label="oscillation amplitude",
        fname="grid_amplitude_beta_mu.png",
    )

    # Optional: mark where max_re is near 0 (candidate boundary)
    # This helps you see the “ridge”
    plt.figure(figsize=(10, 5))
    plt.contour(
        betas,
        mus,
        max_re,
        levels=[0.0],
        linewidths=2,
    )
    plt.xlabel("β")
    plt.ylabel("μ")
    plt.title("Contour where max Re(λ) = 0 (candidate stability boundary)")
    plt.tight_layout()
    plt.savefig("grid_boundary_contour.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

