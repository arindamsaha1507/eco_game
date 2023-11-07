"""Module for plotting the results of the simulation."""

import pylab as pl


def plot_figure(data, filename):
    """Plot the timeseries of the simulation."""

    pl.figure(1)
    pl.plot(data[:, 0], "b", linewidth=1, label="R")
    pl.plot(data[:, 1], "r", linewidth=1, label="P")
    pl.plot(data[:, 2], "g", linewidth=1, label="S")
    pl.ylabel("$R,P,S$", fontsize=15)
    pl.xlabel(r"$time$", fontsize=15)
    pl.legend(frameon=False, fontsize=10)
    pl.rcParams["axes.linewidth"] = 1
    pl.tight_layout()
    pl.savefig(filename)
    # pl.savefig("eco_game_dynamics.pdf")
    # pl.show()
