import scipy.integrate as sp
import numpy as np


def func(t, y):
    return 2*y


print(sp.solve_ivp(func, (0, 10), np.array([1,0,0])).y)