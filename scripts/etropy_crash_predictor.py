import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

def pcc_system(y, t, alpha, beta, gamma):
    P, Co, Ch = y
    dPdt = P * (alpha*Ch - beta*Co)
    dCodt = Co * (beta*P - gamma*Ch)
    dChdt = Ch * (gamma*Co - alpha*P)
    return [dPdt, dCodt, dChdt]

def calculate_entropy(probs):
    probs = np.array(probs)
    probs = probs / (probs.sum() + 1e-9)
    return -np.sum(probs * np.log2(probs + 1e-9))

def run_trial(trial_id):
    alpha = 1.0
    beta = np.random.uniform(0.5, 5.0)
    gamma = np.random.uniform(0.5, 5.0)
    
    t = np.linspace(0, 300, 3000)
    initial_state = [0.33, 0.33, 0.33]
    sol = odeint(pcc_system, initial_state, t, args=(alpha, beta, gamma))
    
    # Identify extinction (any pop < 1%)
    extinction_idx = np.where(np.any(sol < 0.01, axis=1))[0]
    
    if len(extinction_idx) > 0:
        crash_point = extinction_idx[0]
        # Calculate entropy up to the crash
        h_history = [calculate_entropy(row) for row in sol[:crash_point]]
        return True, h_history
    return False, None

# --- Main Execution ---
num_trials = 500
window_size = 150 # How many steps back from the crash to analyze
crash_matrix = []

print(f"Running {num_trials} survival trials with fixed sequence alignment...")

for i in range(num_trials):
    crashed, h_history = run_trial(i)
    if crashed and len(h_history) >= window_size:
        # Grab the last 150 steps before the crash
        crash_matrix.append(h_history[-window_size:])

# --- Visualization ---
plt.figure(figsize=(12, 6))

if crash_matrix:
    crash_matrix = np.array(crash_matrix)
    avg_crash = np.mean(crash_matrix, axis=0)
    std_crash = np.std(crash_matrix, axis=0)
    
    time_axis = np.arange(-window_size, 0)
    plt.plot(time_axis, avg_crash, color='red', lw=3, label='Avg Entropy Leading to Collapse')
    plt.fill_between(time_axis, avg_crash - std_crash, avg_crash + std_crash, color='red', alpha=0.1)

plt.axhline(y=1.0, color='black', linestyle='--', label='Critical Threshold (1.0 bit)')
plt.title("The Entropy Death Rattle: H as a Lead Indicator of Systemic Collapse")
plt.xlabel("Steps Before Extinction (T-0)")
plt.ylabel("Shannon Entropy (H)")
plt.legend()
plt.grid(alpha=0.3)

timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"Entropy_Death_Rattle_{timestamp}.png"
plt.savefig(filename, dpi=300)
print(f"Analysis complete. Result saved as: {filename}")
