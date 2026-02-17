import numpy as np
import matplotlib
matplotlib.use('Agg') # Necessary for servers/environments without a display
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

def pcc_system(y, t, alpha, beta, gamma):
    P, Co, Ch = y
    # Replicator equations tracking population density
    dPdt = P * (alpha*Ch - beta*Co)
    dCodt = Co * (beta*P - gamma*Ch)
    dChdt = Ch * (gamma*Co - alpha*P)
    return [dPdt, dCodt, dChdt]

def check_stability(alpha, beta, gamma):
    t = np.linspace(0, 500, 5000) 
    initial_state = [0.33, 0.33, 0.33]
    sol = odeint(pcc_system, initial_state, t, args=(alpha, beta, gamma))
    
    # Stability threshold: If any strategy drops below 1%, it's a collapse
    if np.any(sol[-100:] < 0.01):
        return 0  # Red Zone: Monoculture
    return 1      # Green Zone: Stable Limit Cycle

# Configuration
res = 50  # Increased resolution for a smoother map
beta_range = np.linspace(0.1, 4.0, res) 
gamma_range = np.linspace(0.1, 4.0, res)
alpha = 1.0 

stability_matrix = np.zeros((res, res))

print(f"Mapping Phase Transitions at {res}x{res} resolution...")
start_time = time.time()

for i, b in enumerate(beta_range):
    for j, g in enumerate(gamma_range):
        stability_matrix[i, j] = check_stability(alpha, b, g)

# Visualization and Saving
plt.figure(figsize=(10, 8))
plt.imshow(stability_matrix, origin='lower', extent=[0.1, 4.0, 0.1, 4.0], 
           cmap='RdYlGn', aspect='auto')
plt.colorbar(label='System State (1=Dynamic Stability, 0=Collapse)')
plt.xlabel('Chaos Strength (Gamma: Ch > Co)')
plt.ylabel('Control Strength (Beta: Co > P)')
plt.title('PCC Stability Simplex: Mapping the Inversion Boundary')

# Annotating the map for the paper
plt.text(0.5, 3.5, "Stability Zone\n(Homeostasis)", color='white', fontweight='bold')
plt.text(3.0, 0.5, "Inversion Zone\n(Collapse)", color='black', fontweight='bold')

# Save logic
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"PCC_Phase_Map_{timestamp}.png"
plt.savefig(filename, dpi=300) # High DPI for publication quality
print(f"Process complete in {time.time() - start_time:.2f} seconds.")
print(f"Figure saved as: {filename}")
