import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider

# --- Configuration ---
N = 100  # Grid size
# 1: Pressure (Blue), 2: Control (Green), 3: Chaos (Red)
# ---------------------

class PCCSimulator:
    def __init__(self, size):
        self.size = size
        self.grid = np.random.randint(1, 4, size=(size, size))
        self.history = {'entropy': []}
        # Success probabilities for P>Ch, Co>P, Ch>Co
        self.strengths = [1.0, 1.0, 1.0] 

    def update(self):
        new_grid = self.grid.copy()
        for _ in range(self.size * self.size // 2): # Sampling half the grid per step for speed
            i, j = np.random.randint(0, self.size), np.random.randint(0, self.size)
            ni, nj = (i + np.random.choice([-1, 0, 1])) % self.size, (j + np.random.choice([-1, 0, 1])) % self.size
            
            attacker, defender = self.grid[ni, nj], self.grid[i, j]
            
            # Interaction Logic with Probability Sliders
            # [0]: P > Ch | [1]: Co > P | [2]: Ch > Co
            if attacker == 1 and defender == 3 and np.random.random() < self.strengths[0]:
                new_grid[i, j] = 1
            elif attacker == 2 and defender == 1 and np.random.random() < self.strengths[1]:
                new_grid[i, j] = 2
            elif attacker == 3 and defender == 2 and np.random.random() < self.strengths[2]:
                new_grid[i, j] = 3
        
        self.grid[:] = new_grid[:]

    def get_entropy(self):
        _, counts = np.unique(self.grid, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-9))

# --- Setup UI ---
sim = PCCSimulator(N)
fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
ax_sliders = [plt.axes([0.7, 0.6 - i*0.1, 0.2, 0.03]) for i in range(3)]

# Plots
cmap = ListedColormap(['#1f77b4', '#2ca02c', '#d62728'])
img = ax1.imshow(sim.grid, cmap=cmap)
ax1.set_title("PCC Spatial Engine")
ax1.axis('off')

line, = ax2.plot([], [], color='purple', lw=2)
ax2.set_xlim(0, 100); ax2.set_ylim(0, 1.6)
ax2.set_title("System Entropy (Homeostasis Level)")

# Sliders
s_p = Slider(ax_sliders[0], 'P Power', 0.1, 1.0, valinit=1.0)
s_co = Slider(ax_sliders[1], 'Co Power', 0.1, 1.0, valinit=1.0)
s_ch = Slider(ax_sliders[2], 'Ch Power', 0.1, 1.0, valinit=1.0)

def animate(frame):
    sim.strengths = [s_p.val, s_co.val, s_ch.val]
    sim.update()
    sim.history['entropy'].append(sim.get_entropy())
    
    img.set_data(sim.grid)
    line.set_data(range(len(sim.history['entropy'])), sim.history['entropy'])
    
    if len(sim.history['entropy']) > ax2.get_xlim()[1]:
        ax2.set_xlim(0, len(sim.history['entropy']) + 50)
    return img, line

ani = animation.FuncAnimation(fig, animate, interval=1, blit=False)
plt.show()
