import argparse
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

@dataclass
class BoxStyle:
    width: float
    height: float
    rounding: float = 0.02
    linewidth: float = 1.5
    fontsize: int = 10

def add_box(ax, x, y, text, style, facecolor="white", edgecolor="black"):
    patch = FancyBboxPatch(
        (x - style.width / 2, y - style.height / 2),
        style.width, style.height,
        boxstyle=f"round,pad=0.02,rounding_size={style.rounding}",
        linewidth=style.linewidth, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=style.fontsize, family="sans-serif")

def add_labeled_arrow(ax, start, end, text=None, side="right", dist=0.02, color="black"):
    """Adds arrow with text shifted to the side of the line."""
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, 
                            linewidth=1.2, color=color, shrinkA=5, shrinkB=5)
    ax.add_patch(arrow)
    
    if text:
        # Calculate midpoint
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        # Determine offset direction (perpendicular to arrow)
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = (dx**2 + dy**2)**0.5
        # Perpendicular vector
        px, py = -dy/length, dx/length
        
        if side == "left": px, py = -px, -py
            
        ax.text(mx + px*dist, my + py*dist, text, fontsize=9, 
                ha="center", va="center", color=color, family="sans-serif")

def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Styles - Reduced size for more 'breathing room'
    param_style = BoxStyle(width=0.14, height=0.07, fontsize=9)
    main_style = BoxStyle(width=0.20, height=0.09, fontsize=11)
    metric_style = BoxStyle(width=0.16, height=0.07, fontsize=9)

    # 1. PARAMETER LAYER (Top)
    add_box(ax, 0.25, 0.85, "Pressure $P$\n(Amplification)", param_style, "#fff2e8")
    add_box(ax, 0.50, 0.90, "Control $C$\n(Stabilization)", param_style, "#eaf4ff")
    add_box(ax, 0.75, 0.85, "Chaos $K$\n(Dispersion)", param_style, "#f4ecff")

    # 2. CORE DYNAMICS (Upper Middle)
    add_box(ax, 0.50, 0.70, "PCC Update Rule\n$x_{t+1} = F(x_t; P, C, K)$", main_style, "#f7f7f7")
    
    # 3. STATE LAYER (Middle)
    add_box(ax, 0.50, 0.52, "Evolving State\n$x(t)$", BoxStyle(0.12, 0.06), "white")

    # 4. OBSERVABLE LAYER (Lower Middle)
    add_box(ax, 0.35, 0.35, "Entropy $S(t)$", metric_style, "#fffdf2")
    add_box(ax, 0.65, 0.35, "Instability $I(t)$", metric_style, "#fffdf2")

    # 5. DERIVED METRICS (Bottom - re-spaced to 4 columns)
    metrics_y = 0.15
    add_box(ax, 0.15, metrics_y, "Entropy Sensitivity\n$dS/dP$", metric_style, "#fff8dc")
    add_box(ax, 0.38, metrics_y, "Critical Pressure\n$P_c$", metric_style, "#fff8dc")
    add_box(ax, 0.62, metrics_y, "Instability Sens.\n$dI/dP$", metric_style, "#fff8dc")
    add_box(ax, 0.85, metrics_y, "Collapse Steepness\n$min(dS/dP)$", metric_style, "#fff8dc")

    # --- ARROWS WITH SIDE LABELS ---
    add_labeled_arrow(ax, (0.30, 0.82), (0.42, 0.74), "drives", side="left")
    add_labeled_arrow(ax, (0.50, 0.86), (0.50, 0.75), "constrains", side="right", dist=0.05)
    add_labeled_arrow(ax, (0.70, 0.82), (0.58, 0.74), "perturbs", side="right")

    # Flow arrows
    add_labeled_arrow(ax, (0.50, 0.65), (0.50, 0.56)) # Dynamics to State
    add_labeled_arrow(ax, (0.45, 0.49), (0.38, 0.39)) # State to Entropy
    add_labeled_arrow(ax, (0.55, 0.49), (0.62, 0.39)) # State to Instability

    # Data Extraction Arrows (Bottom)
    add_labeled_arrow(ax, (0.30, 0.31), (0.18, 0.19))
    add_labeled_arrow(ax, (0.35, 0.31), (0.38, 0.19))
    add_labeled_arrow(ax, (0.65, 0.31), (0.62, 0.19))
    add_labeled_arrow(ax, (0.68, 0.31), (0.82, 0.19))

    plt.title("PCC Simulation & Analysis Pipeline", fontsize=14, fontweight='bold', pad=20)
    plt.savefig("pcc_schematic_fixed.png", dpi=300, bbox_inches="tight")
    print("Saved pcc_schematic_fixed.png")

if __name__ == "__main__":
    main()