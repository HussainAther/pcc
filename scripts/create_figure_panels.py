import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Define the figure configurations based on your file structure
figures_config = {
    "figure2": [
        ("a", "hopf_candidate_boundary.png", "Hopf Bifurcation"),
        ("b", "phase_diagram_schematic.png", "Schematics Phase Diagram"),
        ("c", "hopfcandidateboundary.png", "Hopf Candidate Boundary")
    ],
    "figure3": [
        ("a", "lead_time_dist_eps_0.20.png", "Lead Time Data"),
        ("b", "pooled_lead_time.png", "Pooled Distribution"),
        ("c", "scaling_figure.png", "Log-Log Scaling")
    ],
    "figure4": [
        ("a", "fixationtime.png", "Fixation Time"),
        ("b", "entropy.png", "Entropy")
    ],
    "figure5": [
        ("a", "fixationtimescaling.png", "Scaling"),
        ("b", "fixationtimescaling2.png", "Scaling 2")
    ]
}

def create_combined_figure(fig_name, panels):
    num_panels = len(panels)
    
    # Create a figure layout (1 row for 2-3 panels, or adjust as needed)
    fig = plt.figure(figsize=(5 * num_panels, 5))
    gs = gridspec.GridSpec(1, num_panels)

    for i, (label, filename, title) in enumerate(panels):
        img_path = os.path.join(fig_name, filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found. Skipping.")
            continue
            
        img = Image.open(img_path)
        ax = fig.add_subplot(gs[i])
        
        ax.imshow(img)
        ax.axis('off')
        
        # Add the 'a', 'b', 'c' labels
        ax.text(-0.05, 1.05, label, transform=ax.transAxes, 
                fontsize=20, fontweight='bold', va='top', ha='right')
        
        # Optional: Add the descriptive title
        # ax.set_title(title, fontsize=12)

    plt.tight_layout()
    output_name = f"{fig_name}_combined.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")

# Run the processing
for fig_folder, config in figures_config.items():
    if os.path.exists(fig_folder):
        create_combined_figure(fig_folder, config)
    else:
        print(f"Directory {fig_folder} not found.")
