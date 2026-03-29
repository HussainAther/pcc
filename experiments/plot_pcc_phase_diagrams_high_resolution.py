#!/usr/bin/env python3
import argparse
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCC phase diagrams with Transition Detection")
    parser.add_argument("--phase-table", required=True, help="Path to phase_table.csv")
    parser.add_argument("--summary", required=True, help="Path to summary.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for output plots")
    return parser.parse_args()

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_and_augment_data(phase_table_path: str, summary_path: str) -> pd.DataFrame:
    # Load and merge to ensure all metrics are aligned per (P, C, K) point
    phase_df = pd.read_csv(phase_table_path)
    summary_df = pd.read_csv(summary_path)
    
    # Merge on the coordinates
    df = pd.merge(phase_df, summary_df, on=["pressure", "control", "chaos"], suffixes=('', '_sum'))
    
    # --- CRITICAL: SENSITIVITY CALCULATIONS ---
    # Sort to ensure diff() calculates correctly across pressure steps
    df = df.sort_values(["control", "chaos", "pressure"])
    
    # Group by (Control, Chaos) to find gradients along the Pressure axis
    grouped = df.groupby(["control", "chaos"])
    
    # dS/dP: Entropy Sensitivity
    df["dS_dP"] = grouped["mean_entropy"].diff() / grouped["pressure"].diff()
    
    # dI/dP: Instability Sensitivity
    df["dI_dP"] = grouped["max_instability"].diff() / grouped["pressure"].diff()
    
    # Handle NaNs from diff() (first element of each group)
    df = df.fillna(0)
    
    return df

def sorted_unique(values: pd.Series) -> List[float]:
    return sorted(values.unique().tolist())

def plot_sensitivity_slices(
    df: pd.DataFrame, 
    value_col: str, 
    out_path: str, 
    title: str, 
    cmap: str = "inferno"
):
    """Generates heatmaps of gradients to show phase boundaries."""
    pressure_values = sorted_unique(df["pressure"])
    control_values = sorted_unique(df["control"])
    chaos_values = sorted_unique(df["chaos"])

    # Filter out the first pressure slice if it's all zeros from diff
    plot_pressures = pressure_values[1:] 
    n = len(plot_pressures)
    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    
    # Global scale for sensitivity to compare across slices
    vmin, vmax = df[value_col].min(), df[value_col].max()

    for idx, p in enumerate(plot_pressures):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        
        # Filter slice
        sub = df[np.isclose(df["pressure"], p)]
        matrix = np.zeros((len(control_values), len(chaos_values)))
        
        # Pivot manually to ensure grid alignment
        for i, ctrl in enumerate(control_values):
            for j, chs in enumerate(chaos_values):
                val = sub[(np.isclose(sub["control"], ctrl)) & (np.isclose(sub["chaos"], chs))][value_col]
                matrix[i, j] = val.iloc[0] if not val.empty else 0

        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        
        ax.set_title(f"P = {p:.2f}")
        if r == rows - 1: ax.set_xlabel("Chaos")
        if c == 0: ax.set_ylabel("Control")

    # Clean up empty subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=value_col)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_collapse_curves(df: pd.DataFrame, out_path: str):
    """Plots Entropy vs Pressure for representative points."""
    plt.figure(figsize=(10, 6))
    
    # Pick 5 representative (Control, Chaos) pairs
    ctrls = sorted_unique(df["control"])
    chss = sorted_unique(df["chaos"])
    
    # Samples: corners and center
    samples = [
        (ctrls[0], chss[0]), (ctrls[-1], chss[-1]), 
        (ctrls[len(ctrls)//2], chss[len(chss)//2]),
        (ctrls[0], chss[-1]), (ctrls[-1], chss[0])
    ]
    
    for ctrl, chs in samples:
        sub = df[(np.isclose(df["control"], ctrl)) & (np.isclose(df["chaos"], chs))].sort_values("pressure")
        plt.plot(sub["pressure"], sub["mean_entropy"], marker='o', label=f"C={ctrl:.2f}, K={chs:.2f}")
    
    plt.title("Collapse Curves: Entropy vs Pressure")
    plt.xlabel("Pressure")
    plt.ylabel("Mean Entropy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    ensure_out_dir(args.out_dir)

    print("Loading and calculating gradients...")
    df = load_and_augment_data(args.phase_table, args.summary)

    # 1. dS/dP Heatmaps (Phase Boundaries)
    print("Plotting Entropy Sensitivity (dS/dP)...")
    plot_sensitivity_slices(
        df, "dS_dP", 
        os.path.join(args.out_dir, "sensitivity_entropy.png"),
        "Entropy Sensitivity (dS/dP) - Phase Transitions"
    )

    # 2. dI/dP Heatmaps (Structural Instability)
    print("Plotting Instability Sensitivity (dI/dP)...")
    plot_sensitivity_slices(
        df, "dI_dP", 
        os.path.join(args.out_dir, "sensitivity_instability.png"),
        "Instability Sensitivity (dI/dP)",
        cmap="magma"
    )

    # 3. Collapse Curves
    print("Plotting Collapse Curves...")
    plot_collapse_curves(df, os.path.join(args.out_dir, "collapse_curves.png"))

    print(f"Done. Check {args.out_dir} for results.")

if __name__ == "__main__":
    main()