# phase_diagram_schematic.py
# Paper-style schematic phase diagram with 3 regimes:
# 1) Conservative (neutral cycles)
# 2) Asymmetric (drift / biased cycles / attractor)
# 3) Stochastic fixation (absorbing boundaries due to noise/finite-size/spatiality)

import numpy as np
import matplotlib.pyplot as plt

def make_phase_diagram(
    outpath="phase_diagram_schematic.png",
    # Axis labels (change to match your manuscript)
    x_label=r"Asymmetry strength $\epsilon$",
    y_label=r"Noise / finite-size level $\sigma$",
    title="Schematic phase diagram",
    # Plot ranges
    x_min=0.0, x_max=1.0,
    y_min=0.0, y_max=1.0,
    # Boundary shapes (tune these to match your story)
    # Fixation boundary: sigma > f_fix(eps)
    fix_a=0.18, fix_b=0.55, fix_pow=1.3,
    # Conservative-to-asymmetric boundary: eps > g_asym(sigma) (usually eps > 0)
    asym_eps0=0.10, asym_k=0.20, asym_pow=0.8,
    # Style
    dpi=300,
    show_points=None,  # optional: list of dicts, see example at bottom
):
    """
    Regions (schematic):
      - Conservative regime: low eps (asymmetry) AND low sigma (noise).
      - Asymmetric regime: eps above asym boundary but sigma below fixation boundary.
      - Stochastic fixation: sigma above fixation boundary (dominant), any eps.

    The boundaries are not 'data fits'—they are schematic curves you can tune.
    """

    # Grid for classification
    nx, ny = 600, 600
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xs, ys)

    # Boundary functions (editable)
    def f_fix(eps):
        # Increasing fixation threshold with eps (optional) — tune as needed
        # sigma_fix = a + b * eps^pow
        return fix_a + fix_b * np.power(np.clip(eps, 0, None), fix_pow)

    def g_asym(sig):
        # Asymmetry boundary could mildly increase with noise (optional)
        # eps_asym = eps0 + k * sigma^pow
        return asym_eps0 + asym_k * np.power(np.clip(sig, 0, None), asym_pow)

    # Region codes: 0=Conservative, 1=Asymmetric, 2=Fixation
    sigma_fix = f_fix(X)
    eps_asym = g_asym(Y)

    region = np.zeros_like(X, dtype=int)

    # Fixation dominates
    region[Y >= sigma_fix] = 2

    # Asymmetric: above asym boundary and not fixation
    asym_mask = (X >= eps_asym) & (Y < sigma_fix)
    region[asym_mask] = 1

    # Conservative: everything else (default 0)

    # Build figure
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=dpi)

    # Color map: keep subtle/print-friendly
    # (If your journal is grayscale-heavy, we can switch to hatch patterns.)
    # NOTE: matplotlib wants numeric -> color mapping; use an explicit ListedColormap.
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([
        "#e9f3ff",  # Conservative
        "#fff2cc",  # Asymmetric
        "#f8d7da",  # Fixation
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    ax.imshow(
        region,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )

    # Plot boundaries
    eps_line = np.linspace(x_min, x_max, 600)
    sig_fix_line = f_fix(eps_line)
    ax.plot(eps_line, np.clip(sig_fix_line, y_min, y_max), linewidth=2.0)

    sig_line = np.linspace(y_min, y_max, 600)
    eps_asym_line = g_asym(sig_line)
    ax.plot(np.clip(eps_asym_line, x_min, x_max), sig_line, linewidth=2.0)

    # Labels inside regions
    ax.text(
        x_min + 0.08*(x_max-x_min),
        y_min + 0.12*(y_max-y_min),
        "Conservative\n(neutral cycles)",
        fontsize=11,
        va="center",
        ha="left",
    )
    ax.text(
        x_min + 0.62*(x_max-x_min),
        y_min + 0.25*(y_max-y_min),
        "Asymmetric\n(biased dynamics)",
        fontsize=11,
        va="center",
        ha="center",
    )
    ax.text(
        x_min + 0.55*(x_max-x_min),
        y_min + 0.83*(y_max-y_min),
        "Stochastic fixation\n(absorbing boundaries)",
        fontsize=11,
        va="center",
        ha="center",
    )

    # Optional: overlay points (e.g., your runs)
    # show_points format:
    # [
    #   {"x": [...], "y": [...], "label": "runs", "marker": "o"},
    #   {"x": [...], "y": [...], "label": "special", "marker": "s"},
    # ]
    if show_points:
        for series in show_points:
            ax.scatter(series["x"], series["y"], s=28, marker=series.get("marker", "o"), label=series.get("label", None))
        if any(s.get("label") for s in show_points):
            ax.legend(frameon=True, loc="upper left")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Cleaner frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

    print(f"Saved: {outpath}")


if __name__ == "__main__":
    # Example usage (edit axis names + boundaries)
    make_phase_diagram(
        outpath="phase_diagram_schematic.png",
        x_label=r"Asymmetry strength $\epsilon$",
        y_label=r"Noise level $\sigma$",
        title="Schematic phase diagram of PCC model regimes",
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
        # Tune these three knobs to shape the regime boundaries:
        fix_a=0.18, fix_b=0.50, fix_pow=1.25,
        asym_eps0=0.10, asym_k=0.25, asym_pow=0.75,
        # Optional overlay points:
        show_points=[
            {"x": [0.05, 0.12, 0.20, 0.60], "y": [0.05, 0.10, 0.18, 0.22], "label": "sim runs", "marker": "o"},
            {"x": [0.15, 0.40, 0.80], "y": [0.65, 0.75, 0.90], "label": "fixation observed", "marker": "x"},
        ],
    )

