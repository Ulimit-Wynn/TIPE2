import scipy.integrate as integrate
import numpy as np


class DifferentiableFunction:
    T = 360
    dt = 0.1

    def __init__(self, f=None, vector=None, dim=None, dfdx=None, dfdu=None):
        self.evaluate = f
        self.dx = dfdx
        self.du = dfdu
        self.vector = vector
        self.dim = dim

    def to_vector(self):
        n = np.size(self.evaluate(0))
        v = np.array([self.evaluate(i * self.dt) for i in range(0, int(self.T / self.dt) + 1)])
        v = np.ravel(v)
        self.vector = v
        self.dim = n

    def to_func(self):
        v = np.reshape(self.vector, (-1, self.dim))

        def func(t):
            return v[int(t / self.dt)]

        self.evaluate = func


def gradient(f, g, h):
    def func(t, y):
        return np.atleast_1d(np.atleast_1d(f.dx(t)) @ np.atleast_1d(y) + g.dx(t))

    print('calculating p')
    p_values = integrate.solve_ivp(func, (0, g.T), np.atleast_1d(h.dx(f.T)), t_eval=[i for i in range(0, g.T + 1)]).y
    for i in range(0, np.size(p_values, 0)):
        p_values[i] = p_values[i][::-1]

    def p(t):
        j = int(t * f.T / np.size(p_values[0]))
        return np.array([i[j] for i in p_values])

    def grad_eval(t):
        return np.atleast_1d(np.atleast_1d(f.du(t)) @ p(t) + g.du(t))

    grad = DifferentiableFunction(grad_eval, dfdx=None, dfdu=None)
    grad.to_vector()
    return grad.vector
