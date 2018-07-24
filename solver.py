import scipy.integrate as integrate
import scipy.optimize as optimize
import numpy as np


class DifferentiableFunction:
    T = 360
    dt = 0.1

    def __init__(self, f, vector=None, dim=None, dfdx=None, dfdu=None):
        self.evaluate = f
        self.dx = dfdx
        self.du = dfdu

    def to_vector(self):
        n = np.size(self.evaluate(0))
        v = np.array([self.evaluate(i * self.dt) for i in range(0, int(self.T / self.dt)+1)])
        print(v)
        v = np.ravel(v)
        self.vector = v
        self.dim = n
    def to_func(self):
        v = np.reshape(self.vector, (-1, self.dim))
        def func(t):
            return v[int(t / self.dt)]


def gradient(f, g, h):
    def func(t, y):
        return f.dx(t) @ y + g.dx(f.T)

    p_values = integrate.solve_ivp(func, (0, f.T), h.dx(f.T), t_eval=[i for i in range(0, f.T)]).y
    for i in range(0, np.size(p_values, 0)):
        p_values[i] = p_values[i][::-1]

    def p(t):
        j = int(t * f.T / np.size(p_values[0]))
        return np.array([i[j] for i in p_values])

    def grad_eval(t):
        return f.dx(t) @ p(t)

    grad = DifferentiableFunction(grad_eval, dfdx=None, dfdu=None)
    grad.to_vector()
    return grad.vector


def J(u, g, h):
    return integrate.quad(g.evaluate, 0, DifferentiableFunction.T, args=(u)) + h(DifferentiableFunction.T, u)

print(optimize.minimize(J, u0, args=()))