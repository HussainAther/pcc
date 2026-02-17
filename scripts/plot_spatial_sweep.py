# plot_spatial_sweep.py
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="spatial_sweep.npz")
    ap.add_argument("--out_prefix", type=str, default="fig_spatial")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    eps = data["eps_grid"]
    fix_prob = data["fix_prob"]
    mean_tfix = data["mean_tfix"]
    median_tfix = data["median_tfix"]
    sigma = float(data["sigma"])
    trials = int(data["trials"])

    # Figure 1: Fixation probability
    plt.figure()
    plt.plot(eps, fix_prob, marker="o", linestyle="-")
    plt.xlabel(r"Asymmetry strength $\varepsilon$")
    plt.ylabel("Fixation probability")
    plt.title(f"Fixation probability vs ε (σ={sigma}, trials={trials})")
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_fixprob.png", dpi=250)

    # Figure 2: Fixation time (conditional on fixation)
    plt.figure()
    plt.plot(eps, mean_tfix, marker="o", linestyle="-", label="mean")
    plt.plot(eps, median_tfix, marker="x", linestyle="--", label="median")
    plt.xlabel(r"Asymmetry strength $\varepsilon$")
    plt.ylabel(r"Fixation time $T_{\mathrm{fix}}$ (steps)")
    plt.title(f"Fixation time vs ε (σ={sigma}, conditional on fixation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_tfix.png", dpi=250)

    print(f"Saved: {args.out_prefix}_fixprob.png, {args.out_prefix}_tfix.png")

if __name__ == "__main__":
    main()

