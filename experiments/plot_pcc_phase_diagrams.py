#!/usr/bin/env python3
"""
plot_pcc_phase_diagrams.py

Reads:
- phase_table.csv
- summary.csv

Produces:
- phase_final_entropy.png
- phase_mean_entropy.png
- phase_max_instability.png
- phase_regimes.png

Usage:
    python plot_pcc_phase_diagrams.py \
        --phase-table results_pcc_ebid_full/data/phase_table.csv \
        --summary results_pcc_ebid_full/data/summary.csv \
        --out-dir results_pcc_ebid_full/plots

Notes:
- Assumes a full grid over pressure, control, chaos
- Makes one panel per pressure slice
- x-axis = chaos
- y-axis = control
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCC phase diagrams from simulation CSV outputs")
    parser.add_argument("--phase-table", required=True, help="Path to phase_table.csv")
    parser.add_argument("--summary", required=True, help="Path to summary.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for output plots")
    parser.add_argument(
        "--regime-thresholds",
        nargs=2,
        type=float,
        default=[2.0, 3.3],
        metavar=("LOW_HIGH_BOUNDARY", "MID_HIGH_BOUNDARY"),
        help="Two thresholds for regime classification: low/mid and mid/high. Default: 2.0 3.3",
    )
    return parser.parse_args()


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(phase_table_path: str, summary_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    phase_df = pd.read_csv(phase_table_path)
    summary_df = pd.read_csv(summary_path)

    required_phase_cols = {"pressure", "control", "chaos", "mean_entropy", "final_entropy"}
    required_summary_cols = {
        "pressure",
        "control",
        "chaos",
        "final_entropy",
        "mean_entropy",
        "max_instability",
    }

    missing_phase = required_phase_cols - set(phase_df.columns)
    missing_summary = required_summary_cols - set(summary_df.columns)

    if missing_phase:
        raise ValueError(f"phase_table.csv missing columns: {sorted(missing_phase)}")
    if missing_summary:
        raise ValueError(f"summary.csv missing columns: {sorted(missing_summary)}")

    return phase_df, summary_df


def sorted_unique(values: pd.Series) -> List[float]:
    return sorted(values.unique().tolist())


def make_slice_matrix(
    df: pd.DataFrame,
    pressure_value: float,
    value_col: str,
    control_values: List[float],
    chaos_values: List[float],
) -> np.ndarray:
    """
    Build matrix with rows = control values, cols = chaos values.
    """
    sub = df[np.isclose(df["pressure"], pressure_value)].copy()

    matrix = np.full((len(control_values), len(chaos_values)), np.nan, dtype=float)

    control_to_i = {v: i for i, v in enumerate(control_values)}
    chaos_to_j = {v: j for j, v in enumerate(chaos_values)}

    for _, row in sub.iterrows():
        c = float(row["control"])
        k = float(row["chaos"])
        val = float(row[value_col])
        i = control_to_i[c]
        j = chaos_to_j[k]
        matrix[i, j] = val

    return matrix


def classify_regime(value: float, low_mid: float, mid_high: float) -> int:
    """
    0 = low entropy regime
    1 = mid entropy regime
    2 = high entropy regime
    """
    if value < low_mid:
        return 0
    if value < mid_high:
        return 1
    return 2


def make_regime_matrix(
    df: pd.DataFrame,
    pressure_value: float,
    value_col: str,
    control_values: List[float],
    chaos_values: List[float],
    low_mid: float,
    mid_high: float,
) -> np.ndarray:
    sub = df[np.isclose(df["pressure"], pressure_value)].copy()

    matrix = np.full((len(control_values), len(chaos_values)), np.nan, dtype=float)

    control_to_i = {v: i for i, v in enumerate(control_values)}
    chaos_to_j = {v: j for j, v in enumerate(chaos_values)}

    for _, row in sub.iterrows():
        c = float(row["control"])
        k = float(row["chaos"])
        val = float(row[value_col])
        reg = classify_regime(val, low_mid, mid_high)
        i = control_to_i[c]
        j = chaos_to_j[k]
        matrix[i, j] = reg

    return matrix


def annotate_cells(ax: plt.Axes, matrix: np.ndarray, fmt: str = ".2f", fontsize: int = 8) -> None:
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                fontsize=fontsize,
            )


def plot_pressure_slices(
    df: pd.DataFrame,
    value_col: str,
    out_path: str,
    title_prefix: str,
    cmap: str = "viridis",
    annotate: bool = True,
) -> None:
    pressure_values = sorted_unique(df["pressure"])
    control_values = sorted_unique(df["control"])
    chaos_values = sorted_unique(df["chaos"])

    n = len(pressure_values)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    # global color scale across all pressure slices
    vmin = float(df[value_col].min())
    vmax = float(df[value_col].max())

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), squeeze=False)

    im = None
    for idx, p in enumerate(pressure_values):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        matrix = make_slice_matrix(df, p, value_col, control_values, chaos_values)

        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if annotate:
            annotate_cells(ax, matrix)

        ax.set_title(f"P = {p:.2f}")
        ax.set_xlabel("Chaos")
        ax.set_ylabel("Control")
        ax.set_xticks(range(len(chaos_values)))
        ax.set_xticklabels([f"{v:.2f}" for v in chaos_values])
        ax.set_yticks(range(len(control_values)))
        ax.set_yticklabels([f"{v:.2f}" for v in control_values])

    # hide unused axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.suptitle(title_prefix, fontsize=16)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(value_col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regime_slices(
    df: pd.DataFrame,
    value_col: str,
    out_path: str,
    low_mid: float,
    mid_high: float,
) -> None:
    pressure_values = sorted_unique(df["pressure"])
    control_values = sorted_unique(df["control"])
    chaos_values = sorted_unique(df["chaos"])

    n = len(pressure_values)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    # 0=low, 1=mid, 2=high
    cmap = plt.get_cmap("viridis", 3)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), squeeze=False)

    im = None
    label_map = {
        0: "Low",
        1: "Mid",
        2: "High",
    }

    for idx, p in enumerate(pressure_values):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        matrix = make_regime_matrix(
            df=df,
            pressure_value=p,
            value_col=value_col,
            control_values=control_values,
            chaos_values=chaos_values,
            low_mid=low_mid,
            mid_high=mid_high,
        )

        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
        )

        rows_m, cols_m = matrix.shape
        for i in range(rows_m):
            for j in range(cols_m):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                reg = int(val)
                ax.text(
                    j,
                    i,
                    label_map[reg],
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.set_title(f"P = {p:.2f}")
        ax.set_xlabel("Chaos")
        ax.set_ylabel("Control")
        ax.set_xticks(range(len(chaos_values)))
        ax.set_xticklabels([f"{v:.2f}" for v in chaos_values])
        ax.set_yticks(range(len(control_values)))
        ax.set_yticklabels([f"{v:.2f}" for v in control_values])

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.suptitle(
        f"Entropy Regimes by Pressure Slice\nLow < {low_mid:.2f}, Mid < {mid_high:.2f}, High ≥ {mid_high:.2f}",
        fontsize=16,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Low", "Mid", "High"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir)

    phase_df, summary_df = load_data(args.phase_table, args.summary)

    low_mid, mid_high = args.regime_thresholds

    # Final entropy
    plot_pressure_slices(
        df=phase_df,
        value_col="final_entropy",
        out_path=os.path.join(args.out_dir, "phase_final_entropy.png"),
        title_prefix="Final Entropy Across PCC Pressure Slices",
        cmap="viridis",
        annotate=True,
    )

    # Mean entropy
    plot_pressure_slices(
        df=phase_df,
        value_col="mean_entropy",
        out_path=os.path.join(args.out_dir, "phase_mean_entropy.png"),
        title_prefix="Mean Entropy Across PCC Pressure Slices",
        cmap="viridis",
        annotate=True,
    )

    # Max instability
    plot_pressure_slices(
        df=summary_df,
        value_col="max_instability",
        out_path=os.path.join(args.out_dir, "phase_max_instability.png"),
        title_prefix="Max Instability Across PCC Pressure Slices",
        cmap="magma",
        annotate=True,
    )

    # Regime classes using mean entropy
    plot_regime_slices(
        df=phase_df,
        value_col="mean_entropy",
        out_path=os.path.join(args.out_dir, "phase_regimes.png"),
        low_mid=low_mid,
        mid_high=mid_high,
    )

    print("Saved plots:")
    print(f"- {os.path.join(args.out_dir, 'phase_final_entropy.png')}")
    print(f"- {os.path.join(args.out_dir, 'phase_mean_entropy.png')}")
    print(f"- {os.path.join(args.out_dir, 'phase_max_instability.png')}")
    print(f"- {os.path.join(args.out_dir, 'phase_regimes.png')}")


if __name__ == "__main__":
    main()