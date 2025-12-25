import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 0: Empty, 1: Pressure (P), 2: Control (Co), 3: Chaos (Ch)
N = 100
grid = np.random.randint(1, 4, size=(N, N))

def update(frame, img, grid):
    new_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Check a random neighbor (The 'Encounter')
            ni, nj = (i + np.random.randint(-1, 2)) % N, (j + np.random.randint(-1, 2)) % N
            neighbor = grid[ni, nj]
            me = grid[i, j]
            
            # The PCC Ruleset
            # Co (2) beats P (1) | P (1) beats Ch (3) | Ch (3) beats Co (2)
            if (me == 1 and neighbor == 2) or \
               (me == 2 and neighbor == 3) or \
               (me == 3 and neighbor == 1):
                new_grid[i, j] = neighbor
                
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,

fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='viridis', interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid), frames=200, interval=50)
plt.show()

