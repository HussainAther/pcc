import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

def plot_presentation_3d(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    # Load and setup data
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index='control', columns='chaos', values='critical_pressure_below_1_0')
    
    # Fill NaNs with 0 for a clean base, or handle as needed
    pivot = pivot.fillna(0)
    
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    # Start the figure
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    # 1. THE SURFACE: Use 'viridis' or 'plasma' for high contrast
    surf = ax.plot_surface(X, Y, Z, 
                           cmap='viridis', 
                           edgecolor='none', 
                           alpha=0.9, 
                           antialiased=True,
                           shade=True,
                           rcount=100, ccount=100) # Higher resolution

    # 2. THE PROJECTION: Add contour "shadows" on the bottom (Z=0)
    # This helps viewers map the 3D peaks back to specific C/K coordinates
    cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.3)

    # 3. STYLING: Labels and Ticks
    ax.set_xlabel('\nChaos (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('\nControl (C)', fontsize=12, fontweight='bold')
    ax.set_zlabel('\nCritical Pressure ($P_c$)', fontsize=12, fontweight='bold')
    ax.set_title('PCC Stability Landscape: Phase Transition Surface\n', fontsize=18, pad=20)

    # 4. VIEWPOINT: This angle usually shows the "cliff" best
    ax.view_init(elev=30, azim=225) 

    # 5. CLEANUP: Grid and Colorbar
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.5)

    cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cb.set_label('Threshold Pressure ($P_c$)', fontsize=10)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Presentation-grade surface saved to {out_path}")

# Run with updated path
csv_input = 'results_pcc_v_fine/plots_critical/critical_pressure_summary.csv'
plot_output = 'results_pcc_v_fine/plots_critical/phase_surface_presentation.png'

if __name__ == "__main__":
    plot_presentation_3d(csv_input, plot_output)