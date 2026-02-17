import numpy as np
import matplotlib.pyplot as plt
from simulation import Params, max_realpart_eigs_at_eq

def main():
    sig_vals = np.linspace(0.0, 0.5, 100) 
    mu_vals  = np.linspace(0.0, 0.5, 100)
    Z = np.zeros((len(mu_vals), len(sig_vals)))
    for i, mu in enumerate(mu_vals):
        for j, sig in enumerate(sig_vals):
            Z[i, j] = max_realpart_eigs_at_eq(Params(mu=mu, sigma0=sig, sigma1=0.0, kappa=10.0))

    plt.figure(figsize=(10,6))
    im = plt.imshow(Z, origin="lower", aspect="auto",
                   extent=[sig_vals[0], sig_vals[-1], mu_vals[0], mu_vals[-1]], cmap="RdBu_r")
    plt.colorbar(im, label="max Re(λ)")
    plt.contour(sig_vals, mu_vals, Z, levels=[0.0], colors='black', linewidths=2)
    plt.xlabel("destabilizing strength (σ)")
    plt.ylabel("leakage (μ)")
    plt.title("Hopf Bifurcation: σ vs μ (Re(λ)=0 contour)")
    plt.savefig("hopf_candidate_boundary.png")

if __name__ == "__main__":
    main()
