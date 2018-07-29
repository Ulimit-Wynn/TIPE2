import numpy as np
import scipy.integrate as integrate

T = 1
n = 10
dt = T / n


class DifferentiableFunction:
    def __init__(self, f=None, vector=None, dim=None, dfdx=None, dfdu=None):
        self.evaluate = f
        self.dx = dfdx
        self.du = dfdu
        self.vector = vector
        self.dim = dim

    def to_vector(self):
        dim = np.size(self.evaluate(0))
        v = np.array([(self.evaluate(i * dt) + self.evaluate((i + 1) * dt))/2 for i in range(0, n)])
        v = np.ravel(v)
        self.vector = v
        self.dim = dim

    def to_func(self):
        v = np.reshape(self.vector, (-1, self.dim))

        def func(t):
            if t == T:
                return v[int(T / dt) - 1]
            return v[int(t / dt)]

        self.evaluate = func


class TimeFunction:
    def __init__(self, f=None, vector=None, dim=None):
        self.evaluate = f
        self.vector = vector
        self.dim = dim

    def integrate(self):
        return integrate.quad(self.evaluate, 0, T, limit=100)

    def to_vector(self):
        dim = np.size(self.evaluate(0))
        v = np.array([(self.evaluate(i * dt) + self.evaluate((i + 1) * dt))/2 for i in range(0, n)])
        v = np.ravel(v)
        self.vector = v
        self.dim = dim

    def to_func(self):
        v = np.reshape(self.vector, (-1, self.dim))

        def func(t):
            if t == T:
                return v[int(T / dt) - 1]
            return v[int(t / dt)]

        self.evaluate = func


class DynamicalSystem:
    def __init__(self, f, x0):
        self.f = f
        self.x0 = x0

    def solve(self, u):
        def func(t, x_at_t):
            return self.f(u.evaluate(t), x_at_t)
        solve = integrate.solve_ivp(func, (0, T), self.x0, dense_output=True, rtol=10 ** (-6)).sol
        solution = TimeFunction(f=solve.__call__)
        return solution


class Functional:
    def __init__(self, expression, system, g, h):
        self.evaluate = expression
        self.system = system
        self.g = g
        self.h = h

    def wrap(self, vector):
        u = TimeFunction(vector=vector, dim=int(T / n))
        u.to_func()
        functional = self.evaluate(u, self.system, self.g, self.h)
        functional.to_vector()
        return functional.vector
