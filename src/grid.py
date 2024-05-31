import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.integrate import solve_ivp

# Grid and simulation parameters
size = 50  # Grid size
Du, Dv, Dw = 0.05, 0.05, 0.05  # Movement rates for u, v, w
delta_t = 0.1  # Time step size for the reaction
num_steps = 1000  # Number of simulation steps

# Spatially varying parameters
x = np.linspace(0, 4 * np.pi, size)
y = np.linspace(0, 4 * np.pi, size)
X, Y = np.meshgrid(x, y)
a, b, c, alpha, beta, xi = 0.1, 0.5, 0.25, 0.85, 0.5, 0.25

attack_rate = 0.5
handling_time = 0.5


sig1 = 0.35 * (1 + 0.5 * np.sin(X) * np.cos(Y))
sig2 = 0.95 * (1 + 0.5 * np.sin(X) * np.cos(Y))
sig3 = 0.8 * (1 + 0.5 * np.sin(X) * np.cos(Y))

# Initialize state matrices
u = np.ones((size, size)) * 0.3
v = np.ones((size, size)) * 0.3
w = np.ones((size, size)) * 0.3

# u[np.random.randint(0, size), np.random.randint(0, size)] = 1.0
# v[np.random.randint(0, size), np.random.randint(0, size)] = 1.0
# w[np.random.randint(0, size), np.random.randint(0, size)] = 1.0

# Define Holling type II function
def holling_type_II(prey_density, attack_rate, handling_time):
    """Holling type II functional response."""
    return (attack_rate * prey_density) / (1 + attack_rate * handling_time * prey_density)

# Modify reaction function
def reaction(t, y, a, b, c, sig1, sig2, sig3, alpha, beta, xi, attack_rate, handling_time):
    du = y[0] * (
        -sig1 * y[0] - (a + sig1) * holling_type_II(y[1], attack_rate, handling_time) + (b - sig1) * y[2] + (sig1 + alpha - 1)
    )
    dv = y[1] * (
        (a - sig2) * holling_type_II(y[0], attack_rate, handling_time) - sig2 * y[1] - (c + sig2) * y[2] + (sig2 - beta)
    )
    dw = y[2] * (
        -sig3 * y[2] - (b + sig3) * holling_type_II(y[0], attack_rate, handling_time) + (c - sig3) * y[1] + (sig3 - xi)
    )
    return [du, dv, dw]

# # Define reaction function
# def reaction(t, y, a, b, c, sig1, sig2, sig3, alpha, beta, xi):
#     du = y[0] * (
#         -sig1 * y[0] - (a + sig1) * y[1] + (b - sig1) * y[2] + (sig1 + alpha - 1)
#     )
#     dv = y[1] * ((a - sig2) * y[0] - sig2 * y[1] - (c + sig2) * y[2] + (sig2 - beta))
#     dw = y[2] * (-sig3 * y[2] - (b + sig3) * y[0] + (c - sig3) * y[1] + (sig3 - xi))
#     return [du, dv, dw]


# Movement function
def move_cells(grid, rate):
    new_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] > 0:
                # Randomly choose a direction to move
                moves = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                valid_moves = [
                    (nx, ny)
                    for nx, ny in moves
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]
                ]
                nx, ny = valid_moves[np.random.randint(len(valid_moves))]
                new_grid[nx, ny] += grid[i, j] * rate
                new_grid[i, j] += grid[i, j] * (1 - rate)
    return new_grid


# Simulation loop
for step in range(num_steps):
    print(f"Step {step}")
    # Update each cell with reaction
    for i in range(size):
        for j in range(size):
            y0 = [u[i, j], v[i, j], w[i, j]]
            sol = solve_ivp(
                reaction,
                [0, delta_t],
                y0,
                args=(a, b, c, sig1[i, j], sig2[i, j], sig3[i, j], alpha, beta, xi, attack_rate, handling_time),
                method="RK45",
                t_eval=[delta_t],
            )
            u[i, j], v[i, j], w[i, j] = sol.y[
                :, -1
            ]  # Update with the last computed values

    # Movement step
    u = move_cells(u, Du)
    v = move_cells(v, Dv)
    w = move_cells(w, Dw)

    # Visualization every few steps
    if step % 10 == 0:
        plt.figure(figsize=(10, 10))
        # plt.imshow(u + w + w, cmap="viridis", interpolation="nearest")
        plt.imshow(u + v + w, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        plt.title(f"Step {step}")
        plt.colorbar()
        # plt.show()
        plt.savefig(f"output/output_{step}.png")
        plt.close()
