import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

def pcc_metabolic_system(y, t, alpha, beta, gamma, costs):
    P, Co, Ch = y
    cP, cCo, cCh = costs # Joule-cost per unit of population
    
    # Non-transitive replicator equations + Metabolic Decay
    # We subtract the 'Cost of Maintenance' from the growth rate
    dPdt = P * (alpha*Ch - beta*Co) - (cP * P)
    dCodt = Co * (beta*P - gamma*Ch) - (cCo * Co)
    dChdt = Ch * (gamma*Co - alpha*P) - (cCh * Ch)
    
    return [dPdt, dCodt, dChdt]

# --- Setup Trials ---
t = np.linspace(0, 100, 2000)
initial_state = [0.33, 0.33, 0.33]

# Trial 1: Low Friction (Efficient Information Processing)
low_friction_costs = [0.05, 0.05, 0.05] 
# Trial 2: High Friction (Expensive Control/Bureaucracy)
# We simulate a scenario where 'Control' costs 5x more to maintain than 'Chaos'
high_friction_costs = [0.05, 0.25, 0.05] 

sol_low = odeint(pcc_metabolic_system, initial_state, t, args=(1.0, 1.0, 1.0, low_friction_costs))
sol_high = odeint(pcc_metabolic_system, initial_state, t, args=(1.0, 1.0, 1.0, high_friction_costs))

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Low Friction Plot
ax1.plot(t, sol_low[:, 0], label='Pressure', color='blue')
ax1.plot(t, sol_low[:, 1], label='Control', color='green')
ax1.plot(t, sol_low[:, 2], label='Chaos', color='red')
ax1.set_title("Low Friction (Balanced Metabolic Costs)")
ax1.legend()
ax1.grid(alpha=0.2)

# High Friction Plot
ax2.plot(t, sol_high[:, 0], label='Pressure', color='blue')
ax2.plot(t, sol_high[:, 1], label='Control', color='green')
ax2.plot(t, sol_high[:, 2], label='Chaos', color='red')
ax2.set_title("High Friction (Control Inversion: High Maintenance Cost)")
ax2.set_xlabel("Time")
ax2.legend()
ax2.grid(alpha=0.2)

timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"Metabolic_Friction_{timestamp}.png"
plt.savefig(filename, dpi=300)
print(f"Thermodynamic analysis complete. Result saved as: {filename}")
