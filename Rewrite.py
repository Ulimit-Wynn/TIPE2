import numpy as np
import scipy.integrate as integrate
import sympy
import time


T = 10
n = 100
dt = T / n
grad_time = 0
J_time = 0
to_vector_time = 0
integration_time = 0

def multiply_vectors(a, b):
    A = sympy.Matrix(a).transpose()
    B = sympy.Matrix(b)
    return np.array(A * B).ravel()

def multiply(a, b):
    A = sympy.Matrix(a)
    B = sympy.Matrix(b)
    return np.array(A * B).ravel()


class DifferentiableFunction:
    def __init__(self, f=None, vector=None, dim=None, dfdx=None, dfdu=None):
        self.evaluate = f
        self.dx = dfdx
        self.du = dfdu
        self.vector = vector
        self.dim = dim


class TimeFunction:
    def __init__(self, f=None, vector=None, dim=None):
        self.evaluate = f
        self.vector = vector
        self.dim = dim

    def integrate(self, a, b):
        global integration_time
        start = time.time()
        v = []
        for i in range(0, np.size(self(0))):
            def call_index(t):
                return self(t)[i]

            v.append(integrate.quad(call_index, a, b, epsabs=1e-1, epsrel=1e-1)[0])
        v = np.array(v)
        end = time.time()
        integration_time = integration_time + end - start
        #print("integration time: ", integration_time)
        return v

    def to_vector(self):
        global to_vector_time
        start = time.time()
        dim = np.size(self.evaluate(0))
        v = np.array([(self.integrate(i * T / n, (i + 1) * T / n) * n / T) for i in range(0, n)])
        #v = np.array([((self(i * dt) + self((i + 1) * dt)) / 2) for i in range(0, n)])
        v = np.ravel(v)
        print(type(v))
        self.vector = v
        self.dim = dim
        end = time.time()
        to_vector_time += end - start
        print("vector time: ", to_vector_time)

    def to_func(self):
        v = np.reshape(self.vector, (-1, self.dim))
        def func(t):
            if t == T:
                return v[int(T / dt) - 1]
            return v[int(t / dt)]

        self.evaluate = func

    def __call__(self, t):
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
        def func(time, y):
            return np.atleast_1d(
                multiply(np.atleast_1d(self.system.f.dx(T - time, u(T - time), x(T - time))), np.atleast_1d(y)) + self.g.dx(u(T - time),
                                                                                                                   x(T - time)))

        p_sol = integrate.solve_ivp(func, (0, T), np.atleast_1d(self.h.dx(x(T))), dense_output=True).sol

        def p_eval(time):
            return p_sol.__call__(T - time)

        p = TimeFunction(f=p_eval)

        def grad_eval(time):
            return np.atleast_1d(
                multiply(np.atleast_1d(self.system.f.du(time, u(time), x(time))), np.atleast_1d(p(time))) + np.atleast_1d(
                    self.g.du(u(time), x(time))))

        grad = TimeFunction(grad_eval)
        end = time.time()
        return grad

    def grad_vector(self, u):
        u.to_func()
        grad = self.grad(u)
        grad.to_vector()
        grad.vector = T / n * grad.vector
        return grad.vector

    def grad_wrapper(self, vector):
        global grad_time
        start = time.time()
        u = TimeFunction(vector=vector, dim=int(np.size(vector) / n))
        grad = self.grad_vector(u)
        end = time.time()
        grad_time = grad_time + end - start
        #print("grad time: ", grad_time)
        return grad

    def J_wrapper(self, vector):
        global J_time
        start = time.time()
        u = TimeFunction(vector=vector, dim=int(np.size(vector) / n))
        J = self(u)
        end = time.time()
        J_time = J_time + end - start
        #print("J time: ", J_time)
        return J
