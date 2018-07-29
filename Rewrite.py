import numpy as np
import scipy.integrate as integrate

T = 56
n = 100
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
        v = np.array([(self.evaluate(i * dt) + self.evaluate((i + 1) * dt)) / 2 for i in range(0, n)])
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
        v = np.array([(integrate.quad(self.evaluate, i * T / n, (i + 1) * T / n, limit=100)[0])*n / T for i in range(0, n)])
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

    def __call__(self, t, args=None):
        return self.evaluate(t)


class DynamicalSystem:
    def __init__(self, f, x0):
        self.f = f
        self.x0 = x0

    def solve(self, u):
        def func(t, x_at_t):
            return self.f.evaluate(t, u.evaluate(t), x_at_t)

        solve = integrate.solve_ivp(func, (0, T), self.x0, dense_output=True, rtol=10 ** (-13), atol=10 ** (-8)).sol
        solution = TimeFunction(f=solve.__call__)
        return solution


class Functional:
    def __init__(self, system, g, h):
        self.system = system
        self.g = g
        self.h = h

    def __call__(self, u):
        u.to_func()
        x = self.system.solve(u)

        def g_integrable(t):
            return self.g.evaluate(u(t), x(t))

        j = integrate.quad(g_integrable, 0, T)[0] + self.h.evaluate(x(T))
        return j

    def grad(self, u):
        x = self.system.solve(u)

        def func(t, y):
            return np.atleast_1d(
                np.atleast_1d(self.system.f.dx(T - t, u(T - t), x(T - t))) @ np.atleast_1d(y) + self.g.dx(u(T - t),
                                                                                                   x(T - t)))

        p_sol = integrate.solve_ivp(func, (0, T), np.atleast_1d(self.h.dx(x(T))), dense_output=True).sol

        def p_eval(t):
            return p_sol.__call__(T - t)

        p = TimeFunction(f=p_eval)

        def grad_eval(t):
            return np.atleast_1d(
                np.atleast_1d(self.system.f.du(t, u(t), x(t))) @ np.atleast_1d(p(t)) + np.atleast_1d(
                    self.g.du(u(t), x(t))))

        grad = TimeFunction(grad_eval)
        return grad

    def grad_vector(self, u):
        u.to_func()
        grad = self.grad(u)
        grad.to_vector()
        grad.vector = T/n * grad.vector
        return grad.vector

    def grad_wrapper(self, vector):
        u = TimeFunction(vector=vector, dim= int(np.size(vector)/n))
        return self.grad_vector(u)

    def J_wrapper(self, vector):
        u = TimeFunction(vector=vector, dim=int(np.size(vector)/n))
        return self(u)
