import scipy.integrate as sp
import numpy as np
from solver import DifferentiableFunction


def f_eval(t):
    return np.array([2 * t, t-1])

f = DifferentiableFunction(f_eval)
f.to_vector()

print(f.vector)
print(f.evaluate(5))
f.vector = 2 * f.vector
print(f.evaluate(5))