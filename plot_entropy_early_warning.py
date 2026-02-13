# plot_entropy_early_warning.py
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

def smooth(y: np.ndarray, w: int = 11) -> np.ndarray:
    if w <= 1:
        return y
    w = int(w)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="spatial_sweep.npz")
    ap.add_argument("--eps", type=float, required=True, help="which epsilon value to plot")
    ap.add_argument("--out", type=str, default="fig_entropy_warning.png")
    ap.add_argument("--smooth_w", type=int, default=15)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    traces = data["example_traces"].item()

    # Find the closest epsilon key
    keys = np.array([float(k) for k in traces.keys()])
    k = float(keys[np.argmin(np.abs(keys - args.eps))])
    runs = traces[str(k)]
    if len(runs) == 0:
        raise RuntimeError("No traces stored for that epsilon")

    plt.figure()
    for (t, H, p) in runs:
        # Estimate derivative dH/dt on sampled grid
        dH = np.gradient(H, t)
        dH = smooth(dH, args.smooth_w)
        # Plot against time
        plt.plot(t, dH, linestyle="-", alpha=0.8)

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("time (steps)")
    plt.ylabel(r"entropy rate $\dot S(t)$ (nats/step)")
    plt.title(rf"Entropy-rate traces (ε={k:.3f})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=250)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()

