import numpy as np
import matplotlib.pyplot as plt
from simulation import Params, simulate

def project_ternary(traj):
    """
    Transforms 3D strategy shares (x1, x2, x3) into 2D Cartesian coordinates (u, v)
    for plotting on an equilateral triangle.
    """
    x1, x2, x3 = traj[:, 0], traj[:, 1], traj[:, 2]
    # Mapping to 2D: S1 at (0,0), S2 at (1,0), S3 at (0.5, sqrt(3)/2)
    u = x2 + 0.5 * x3
    v = (np.sqrt(3)/2) * x3
    return u, v

def draw_simplex_boundary(ax):
    """Draws the equilateral triangle representing the strategy space."""
    boundary = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(boundary[:, 0], boundary[:, 1], 'k-', lw=1.5)
    ax.text(-0.05, -0.05, "S1", fontsize=12)
    ax.text(1.02, -0.05, "S2", fontsize=12)
    ax.text(0.48, np.sqrt(3)/2 + 0.02, "S3", fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')

def main():
    # Define parameters for both regimes
    p_stable = Params(mu=0.3, sigma0=0.1, sigma1=0.0, kappa=10.0, use_entropy=False)
    p_hopf   = Params(mu=0.1, sigma0=0.4, sigma1=0.0, kappa=10.0, use_entropy=False)
    
    # Simulate trajectories starting slightly away from center
    _, y_stable = simulate(params=p_stable, t_span=(0, 300), y0=(0.4, 0.3, 0.3))
    _, y_hopf = simulate(params=p_hopf, t_span=(0, 500), y0=(0.35, 0.35, 0.3))

    u_s, v_s = project_ternary(y_stable)
    u_h, v_h = project_ternary(y_hopf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Stable spiral toward the interior equilibrium
    draw_simplex_boundary(ax1)
    ax1.plot(u_s, v_s, 'b-', alpha=0.6, lw=1)
    ax1.plot(u_s[0], v_s[0], 'go', label="Start")
    ax1.plot(u_s[-1], v_s[-1], 'rx', label="End")
    ax1.set_title(f"Stable Spiral ($\mu$={p_stable.mu}, $\sigma$={p_stable.sigma})")

    # Plot 2: Outward spiral to stable limit cycle
    draw_simplex_boundary(ax2)
    ax2.plot(u_h, v_h, 'r-', alpha=0.4, lw=0.8)
    # Highlight the final limit cycle orbit
    ax2.plot(u_h[-1500:], v_h[-1500:], 'k-', lw=2, label="Limit Cycle")
    ax2.plot(u_h[0], v_h[0], 'go')
    ax2.set_title(f"Supercritical Hopf Bifurcation ($\mu$={p_hopf.mu}, $\sigma$={p_hopf.sigma})")

    plt.tight_layout()
    plt.savefig("ternary_phase_portrait.png", dpi=300)

if __name__ == "__main__":
    main()
