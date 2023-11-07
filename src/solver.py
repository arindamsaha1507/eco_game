import numpy as np
import yaml
from scipy.integrate import ode

from src.function import ode_func


def run_simulation():
    tf = 100_000
    t = np.linspace(0, tf, tf * 10)

    with open("configuration/settings.yml", "r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    p = settings["parameters"]
    p = list(p.values())
    y0 = settings["initial_conditions"]
    y0 = np.array(list(y0.values()))
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0

    r = ode(ode_func).set_integrator("vode", method="Adams")
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

    return y
