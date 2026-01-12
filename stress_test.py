import numpy as np
import matplotlib.pyplot as plt
from simulation import pcc_system
from scipy.integrate import odeint

def run_experiment(control_strength):
    # P, Co, Ch
    initial_state = [0.33, 0.33, 0.33]
    t = np.linspace(0, 100, 2000)
    
    # We vary 'beta' (the strength of Control over Pressure)
    params = (1.0, control_strength, 1.0)
    solution = odeint(pcc_system, initial_state, t, args=params)
    
    # Check if any population hits near-zero (Extinction)
    extinction = np.any(solution < 0.01)
    return extinction, solution

# Test a range of 'Control' power levels
strengths = np.linspace(0.5, 5.0, 20)
results = [run_experiment(s)[0] for s in strengths]

plt.figure(figsize=(8, 4))
plt.plot(strengths, results, 'ro-')
plt.title("System Collapse vs. Control Strength")
plt.xlabel("Control Dominance (Beta)")
plt.ylabel("Extinction Occurred (1=Yes, 0=No)")
plt.grid(True)
plt.show()
