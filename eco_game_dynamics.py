# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:30:23 2023

@author: subrata
"""

import numpy as np

# from numpy import *

# from scipy import stats
# from scipy.integrate import odeint,  solve_ivp
import pylab as pl
import yaml

# import random
# import math
# import scipy.io
# import networkx as nx
from scipy.integrate import ode


def f(t, y, p):
    """Function to be integrated."""
    a, b, c, sig1, sig2, sig3, alpha, beta, xi = p
    u, v, w = y

    dy = np.empty(3, float)

    dy[0] = u * (-sig1 * u - (a + sig1) * v + (b - sig1) * w + (sig1 + alpha - 1))
    dy[1] = v * ((a - sig2) * u - sig2 * v - (c + sig2) * w + (sig2 - beta))
    dy[2] = w * (-sig3 * w - (b + sig3) * u + (c - sig3) * v + (sig3 - xi))

    return dy


tf = 100_000
t = np.linspace(0, tf, tf * 10)
T = []

with open("configuration/settings.yml", "r", encoding="utf-8") as file:
    settings = yaml.safe_load(file)

p = settings["parameters"]
p = list(p.values())
y0 = settings["initial_conditions"]
y0 = np.array(list(y0.values()))
y = np.zeros((len(t), len(y0)))
y[0, :] = y0

r = ode(f).set_integrator("vode", method="Adams")
# r = ode(f).set_integrator('zvode',method='bdf')
r.set_f_params(p).set_initial_value(y0, t[0])

with open("output.csv", "w", encoding="utf-8") as file:
    file.write("time,R,P,S\n")
    for j in range(1, len(t)):
        # print(j)
        y[j, :] = r.integrate(t[j])
        y0 = y[j, :]
        r.set_initial_value(y0, t[j])
        if j > 0.999 * len(t):
            file.write(f"{t[j]},{y[j,0]},{y[j,1]},{y[j,2]}\n")


pl.figure(1)
pl.plot(y[:, 0], "b", linewidth=1, label="R")
pl.plot(y[:, 1], "r", linewidth=1, label="P")
pl.plot(y[:, 2], "g", linewidth=1, label="S")
pl.ylabel("$R,P,S$", fontsize=15)
pl.xlabel(r"$time$", fontsize=15)
# pl.xticks(fontsize=20)
# pl.yticks(fontsize=20)
pl.legend(frameon=False, fontsize=10)
pl.rcParams["axes.linewidth"] = 1
pl.tight_layout()
pl.savefig("eco_game_dynamics.pdf")
# pl.show()
