import scipy.integrate as integrate
import numpy as np


class DifferentiableFunction:
    T = 100
    n = 1000
    dt = T / n
    x0 = 1

    def __init__(self, f=None, vector=None, dim=None, dfdx=None, dfdu=None):
        self.evaluate = f
        self.dx = dfdx
        self.du = dfdu
        self.vector = vector
        self.dim = dim

    def to_vector(self):
        dim = np.size(self.evaluate(0))
        v = np.array([(self.evaluate(i * self.dt) + self.evaluate((i + 1) * self.dt))/2 for i in range(0, self.n)])
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


def J(u, f, g, h):
    def func(t, x_t):
        u_t = u.evaluate(t)
        return f.evaluate(u_t, x_t)
    x_solve = integrate.solve_ivp(func, (0, f.T), np.array([g.x0]), dense_output=True).sol

    def x(t):
        return x_solve.__call__(t)

    def g_integratable(t):
        return g.evaluate(u.evaluate(t), x(t))

    j = integrate.quad(g_integratable, 0, g.T)[0] + h.evaluate(u.evaluate(g.T), x(g.T))
    return j

def gradient(u, f, g, h):
    def func(t, x):
        return f.evaluate(u.evaluate(t), x)
    x_solve = integrate.solve_ivp(func, (0, f.T), np.array([g.x0]), dense_output=True).sol

    def x(t):
        return x_solve.__call__(t)

    def func(t, y):
        return np.atleast_1d(np.atleast_1d(f.dx(u.evaluate(f.T - t), x(f.T - t))) @ np.atleast_1d(y) + g.dx(u.evaluate(f.T - t), x(f.T - t)))
    print("Calculating p")

    p_sol = integrate.solve_ivp(func, (0, g.T), np.atleast_1d(h.dx(u.evaluate(f.T), x(f.T))), dense_output=True).sol

    def p_eval(t):
        return p_sol.__call__(g.T-t)
    print("p calculated")
    p = DifferentiableFunction(f=p_eval)
    def grad_eval(t):
        return np.atleast_1d(np.atleast_1d(f.du(u.evaluate(t), x(t))) @ np.atleast_1d(p.evaluate(t)) + np.atleast_1d(g.du(u.evaluate(t), x(t))))

    grad = DifferentiableFunction(grad_eval, dfdx=None, dfdu=None)
    return grad
