import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
N = 100  # Grid size (NxN)
# 1: Pressure (Blue), 2: Control (Green), 3: Chaos (Red)
# ---------------------

# Initialize grid with random states
grid = np.random.randint(1, 4, size=(N, N))

def update(frame, img, grid):
    new_grid = grid.copy()
    
    # We use a Monte Carlo step approach for statistical mechanical accuracy
    for _ in range(N * N):
        # Pick a random cell
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        
        # Pick a random neighbor (Von Neumann neighborhood)
        ni, nj = i, j
        if np.random.random() < 0.5:
            ni = (i + np.random.choice([-1, 1])) % N
        else:
            nj = (j + np.random.choice([-1, 1])) % N
            
        attacker = grid[ni, nj]
        defender = grid[i, j]
        
        # PCC Interaction Logic:
        # Pressure (1) beats Chaos (3)
        # Control (2) beats Pressure (1)
        # Chaos (3) beats Control (2)
        if (attacker == 1 and defender == 3) or \
           (attacker == 2 and defender == 1) or \
           (attacker == 3 and defender == 2):
            new_grid[i, j] = attacker
                
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,

# Setup Visualization
fig, ax = plt.subplots(figsize=(8, 8))
# Custom colormap: 1=Blue (P), 2=Green (Co), 3=Red (Ch)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#1f77b4', '#2ca02c', '#d62728'])

img = ax.imshow(grid, cmap=cmap, interpolation='nearest')
ax.set_title("PCC Spatial Dynamics: Emergent Spiral Waves")
ax.axis('off')

ani = animation.FuncAnimation(fig, update, fargs=(img, grid), frames=200, interval=1, blit=True)
plt.show()
