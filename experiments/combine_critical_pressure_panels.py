import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def create_combined_figure(csv_path, out_dir):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Setup plot - 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    
    thresholds = [2.0, 1.0]
    panel_labels = ['(a)', '(b)']
    
    # We want a shared color scale for easier comparison between the two
    # Or keep them independent if the ranges vary wildly. 
    # Here we'll use a shared range based on the pressure values (0.0 to 1.0)
    vmin, vmax = 0.0, 1.0
    
    for i, thr in enumerate(thresholds):
        ax = axes[i]
        col_name = f"critical_pressure_below_{str(thr).replace('.', '_')}"
        
        # Pivot the data
        pivot = df.pivot(index='control', columns='chaos', values=col_name)
        
        # Plot Heatmap
        im = ax.imshow(pivot, origin='lower', aspect='auto', 
                       cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Panel Formatting
        ax.set_title(f"Critical Pressure at $\\tau = {thr}$", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Chaos ($K$)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Control ($C$)", fontsize=12)
            
        # Set ticks to actual parameter values
        x_ticks = np.linspace(0, len(pivot.columns)-1, 5).astype(int)
        y_ticks = np.linspace(0, len(pivot.index)-1, 5).astype(int)
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{pivot.columns[t]:.2f}" for t in x_ticks])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{pivot.index[t]:.2f}" for t in y_ticks])
        
        # Add Panel Label (a, b)
        ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')

    # Add a single shared colorbar to keep it clean
    fig.subplots_adjust(right=0.88, wspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Critical Pressure ($P_c$)', fontsize=12, fontweight='bold')

    # Save
    save_path = os.path.join(out_dir, "fig_critical_pressure_composite.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight') # PDF for the paper
    print(f"Composite figure saved to: {save_path}")

if __name__ == "__main__":
    csv_input = 'results_pcc_v_fine/plots_critical/critical_pressure_summary.csv'
    output_directory = 'results_pcc_v_fine/plots_critical'
    create_combined_figure(csv_input, output_directory)