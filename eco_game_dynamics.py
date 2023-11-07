# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:30:23 2023

@author: subrata
"""

from src.solver import run_simulation
from src.plotter import plot_figure

if __name__ == "__main__":
    data = run_simulation()
    plot_figure(data, "eco_game_dynamics.pdf")
