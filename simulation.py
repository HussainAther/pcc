import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pcc_system(y, t, alpha, beta, gamma):
    """
    P: Pressure (Mass/Force)
    Co: Control (Information/Precision)
    Ch: Chaos (Entropy/Stochasticity)
    """
    P, Co, Ch = y
    
    # Non-transitive competition equations
    # Pressure eats Chaos, Control eats Pressure, Chaos eats Control
    dPdt = P * (alpha*Ch - beta*Co)
    dCodt = Co * (beta*P - gamma*Ch)
    dChdt = Ch * (gamma*Co - alpha*P)
    
    return [dPdt, dCodt, dChdt]

# Initial populations (balanced)
initial_state = [0.33, 0.33, 0.33]
t = np.linspace(0, 50, 1000)

# Interaction Strengths (The 'Laws' of your system)
# alpha: P beats Ch | beta: Co beats P | gamma: Ch beats Co
params = (1.0, 1.0, 1.0)

# Run Simulation
solution = odeint(pcc_system, initial_state, t, args=params)

# Plotting the 'Metabolic' Rhythm of the System
plt.figure(figsize=(10, 5))
plt.plot(t, solution[:, 0], label='Pressure (Force)', color='blue')
plt.plot(t, solution[:, 1], label='Control (Information)', color='green')
plt.plot(t, solution[:, 2], label='Chaos (Stochasticity)', color='red')
plt.title("PCC Dynamics: Evolutionary Stable Strategy (ESS)")
plt.xlabel("Time")
plt.ylabel("Population Density / Influence")
plt.legend()
plt.grid(True)
plt.show()
