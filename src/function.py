"""ODE Function to be integrated."""

import numpy as np


def ode_func(_, y, p):
    """Rock Paper Scissors ODE function."""

    a, b, c, sig1, sig2, sig3, alpha, beta, xi = p
    u, v, w = y

    dy = np.empty(3, float)

    dy[0] = u * (-sig1 * u - (a + sig1) * v + (b - sig1) * w + (sig1 + alpha - 1))
    dy[1] = v * ((a - sig2) * u - sig2 * v - (c + sig2) * w + (sig2 - beta))
    dy[2] = w * (-sig3 * w - (b + sig3) * u + (c - sig3) * v + (sig3 - xi))

    return dy
