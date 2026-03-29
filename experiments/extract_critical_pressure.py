#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Extract critical pressure maps from PCC phase data")
    parser.add_argument("--phase-table", required=True, help="Path to phase_table.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for outputs")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2.0, 1.0],
                        help="Entropy thresholds for critical pressure detection")
    return parser.parse_args()


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def first_crossing_pressure(group: pd.DataFrame, threshold: float):
    group = group.sort_values("pressure")
    crossed = group[group["mean_entropy"] < threshold]
    if crossed.empty:
        return np.nan
    return float(crossed.iloc[0]["pressure"])


def collapse_steepness(group: pd.DataFrame):
    group = group.sort_values("pressure").copy()
    dS = group["mean_entropy"].diff()
    dP = group["pressure"].diff()
    slope = dS / dP
    slope = slope.replace([np.inf, -np.inf], np.nan).dropna()
    if slope.empty:
        return np.nan
    return float(slope.min())  # most negative slope


def make_matrix(df_map: pd.DataFrame, value_col: str):
    control_vals = sorted(df_map["control"].unique())
    chaos_vals = sorted(df_map["chaos"].unique())

    matrix = np.full((len(control_vals), len(chaos_vals)), np.nan)

    for i, c in enumerate(control_vals):
        for j, k in enumerate(chaos_vals):
            sub = df_map[(np.isclose(df_map["control"], c)) & (np.isclose(df_map["chaos"], k))]
            if not sub.empty:
                matrix[i, j] = sub.iloc[0][value_col]

    return matrix, control_vals, chaos_vals


def plot_heatmap(matrix, control_vals, chaos_vals, title, cbar_label, out_path, cmap="viridis", annotate=True):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)

    plt.xticks(range(len(chaos_vals)), [f"{x:.2f}" for x in chaos_vals])
    plt.yticks(range(len(control_vals)), [f"{y:.2f}" for y in control_vals])
    plt.xlabel("Chaos")
    plt.ylabel("Control")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    ensure_out_dir(args.out_dir)

    df = pd.read_csv(args.phase_table)
    df = df.sort_values(["control", "chaos", "pressure"])

    grouped = df.groupby(["control", "chaos"], as_index=False)

    results = []

    for (control, chaos), group in grouped:
        row = {
            "control": float(control),
            "chaos": float(chaos),
            "collapse_steepness": collapse_steepness(group),
        }
        for thr in args.thresholds:
            row[f"critical_pressure_below_{str(thr).replace('.', '_')}"] = first_crossing_pressure(group, thr)
        results.append(row)

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "critical_pressure_summary.csv")
    out_df.to_csv(out_csv, index=False)

    # Plot one heatmap per threshold
    for thr in args.thresholds:
        col = f"critical_pressure_below_{str(thr).replace('.', '_')}"
        matrix, control_vals, chaos_vals = make_matrix(out_df, col)
        plot_heatmap(
            matrix=matrix,
            control_vals=control_vals,
            chaos_vals=chaos_vals,
            title=f"Critical Pressure Map: First P where Mean Entropy < {thr}",
            cbar_label="Critical Pressure",
            out_path=os.path.join(args.out_dir, f"critical_pressure_below_{str(thr).replace('.', '_')}.png"),
            cmap="viridis",
            annotate=True,
        )

    # Plot collapse steepness
    matrix, control_vals, chaos_vals = make_matrix(out_df, "collapse_steepness")
    plot_heatmap(
        matrix=matrix,
        control_vals=control_vals,
        chaos_vals=chaos_vals,
        title="Collapse Steepness Map (Most Negative dS/dP)",
        cbar_label="Min dS/dP",
        out_path=os.path.join(args.out_dir, "collapse_steepness.png"),
        cmap="magma",
        annotate=True,
    )

    print("Done.")
    print(f"Saved summary CSV: {out_csv}")
    for thr in args.thresholds:
        print(os.path.join(args.out_dir, f"critical_pressure_below_{str(thr).replace('.', '_')}.png"))
    print(os.path.join(args.out_dir, "collapse_steepness.png"))


if __name__ == "__main__":
    main()