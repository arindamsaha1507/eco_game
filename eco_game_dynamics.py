"""Main module for the eco_game_dynamics package."""

from src.solver import run_simulation
from src.plotter import plot_figure

if __name__ == "__main__":
    data = run_simulation()
    plot_figure(data, "eco_game_dynamics.pdf")
