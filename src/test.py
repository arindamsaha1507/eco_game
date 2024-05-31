import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
lambda_param = 3.5
epsilon = 0.35
size = 200  # Grid size
steps = 10_000  # Number of iterations

# Initialize the grid
grid = np.random.rand(size, size)

def logistic_map(x, lambda_param):
    return lambda_param * x * (1 - x)

def update_grid(grid, lambda_param, epsilon):
    new_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            # Periodic boundary conditions
            left = grid[i, j-1]
            right = grid[i, (j+1) % size]
            up = grid[i-1, j]
            down = grid[(i+1) % size, j]
            
            # Applying the logistic coupled map lattice formula
            new_grid[i, j] = ((1 - epsilon) * logistic_map(grid[i, j], lambda_param) +
                              (epsilon/4) * (logistic_map(left, lambda_param) +
                                             logistic_map(right, lambda_param) +
                                             logistic_map(up, lambda_param) +
                                             logistic_map(down, lambda_param)))
        
    return new_grid

# Simulation
fig, ax = plt.subplots()
for _ in range(steps):
    print(f"Step {_}")
    grid = update_grid(grid, lambda_param, epsilon)

    pd.DataFrame(grid).to_csv(f"output/output_data_{_}.csv")


ax.clear()
ax.imshow(grid, cmap='hot')
ax.set_title("Step " + str(_))
# plt.pause(0.1)


plt.savefig("output.png")

# plt.show()
