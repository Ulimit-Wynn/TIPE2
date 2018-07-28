import scipy.integrate as integrate
import numpy as np


class DifferentiableFunction:
    T = 1
    n = 100
    dt = T / n

    def __init__(self, f=None, vector=None, dim=None, dfdx=None, dfdu=None):
        self.evaluate = f
        self.dx = dfdx
        self.du = dfdu
        self.vector = vector
        self.dim = dim

    def to_vector(self):
        dim = np.size(self.evaluate(0))
        v = np.array([self.evaluate(i * self.dt) + self.evaluate((i + 1) * self.dt) for i in range(0, self.n)])
        v = np.ravel(v)
        self.vector = v
        self.dim = dim

    def to_func(self):
        v = np.reshape(self.vector, (-1, self.dim))

        def func(t):
            if t == self.T:
                return v[int(self.T / self.dt) - 1]
            return v[int(t / self.dt)]

        self.evaluate = func


def gradient(u, x, f, g, h):
    def func(t, y):
        return np.atleast_1d(np.atleast_1d(f.dx(t, u, x)) @ np.atleast_1d(y) + g.dx(t, u, x))
    print("Calculating p")

    p_sol = integrate.solve_ivp(func, (0, g.T), np.atleast_1d(h.dx(f.T, u, x)),dense_output=True).sol
    def p_eval(t):
        return p_sol.__call__(g.T-t)
    print("p calculated")
    p = DifferentiableFunction(f=p_eval)
    def grad_eval(t):
        return np.atleast_1d(np.atleast_1d(f.du(t, u, x)) @ np.atleast_1d(p.evaluate(t)) + np.atleast_1d(g.du(t, u, x)))

    grad = DifferentiableFunction(grad_eval, dfdx=None, dfdu=None)
    grad.to_vector()
    return grad.vector
