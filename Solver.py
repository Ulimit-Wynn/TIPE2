import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import time
from Variables import *


to_vector_time = 0
integration_time = 0
solving_time = 0



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

    def integrate_v(self, a, b):
        global integration_time
        start = time.time()
        v = []
        for i in range(0, np.size(self(0))):
            def call_index(t):
                return self(t)[i]

            v.append(integrate.quad(call_index, a, b, epsabs=1e-14, epsrel=1e-14)[0])
        v = np.array(v)
        end = time.time()
        integration_time = integration_time + end - start
        # print("integration time: ", integration_time)
        return v

    def to_vector(self, step=n, period=T):
        global to_vector_time
        start = time.time()
        dt = period / step
        dim = np.size(self.evaluate(0))
        v1 = np.array([1 / dt * TimeFunction.integrate_v(self, i * dt, (i + 1) * dt) for i in range(0, step)])
        v = np.array([sum([W_i[j] * self(dt * (i + X_i[j])) for j in range(deg)])
                      for i in range(0, step)])
        v = v.ravel()
        v1 = v1.ravel()
        self.vector = v
        self.dim = dim
        end = time.time()
        to_vector_time += end - start
        # print(to_vector_time)

    def to_func(self):
        v = np.reshape(self.vector, (-1, self.dim))

        def func(t):
            if t == T:
                return v[int(T / dt) - 1]
            if t > T or t < 0:
                return np.zeros(self.dim)
            return v[int(t / dt)]

        self.evaluate = func

    def __call__(self, t):
        return self.evaluate(t)


class DynamicalSystem:
    def __init__(self, f, x0):
        self.f = f
        self.x0 = x0

    def solve(self, u):
        global solving_time
        start = time.time()

        def func(t, x_at_t):
            dx = self.f.evaluate(t, u.evaluate(t), x_at_t)
            return dx

        solve = integrate.solve_ivp(func, (0, T), self.x0, dense_output=True, atol=1e-12, rtol=1e-12).sol
        solution = TimeFunction(f=solve.__call__)
        end = time.time()
        solving_time += end - start
        return solution


class Functional:
    def __init__(self, system, g, h):
        self.system = system
        self.g = g
        self.h = h

    def __call__(self, u):
        u.to_func()
        x = self.system.solve(u)
        if self.g == 0:
            return self.h.evaluate(x(T))

        def g_integrable(t):
            return self.g.evaluate(u(t), x(t))

        j = integrate.quad(g_integrable, 0, T, epsrel=1e-14, epsabs=1e-14)[0] + self.h.evaluate(x(T))
        print("h: ", self.h.evaluate(x(T)))
        print("J: ", j)
        return j

    def grad(self, u, plots=None):
        x = self.system.solve(u)
        global grad_time
        start = time.time()
        if self.g == 0:
            def func(t, y):
                return np.atleast_1d(
                    np.atleast_1d(self.system.f.dx(T - t, u(T - t), x(T - t)).transpose()) @
                    np.atleast_1d(y))

        else:
            def func(t, y):
                return np.atleast_1d(
                    np.atleast_1d(self.system.f.dx(T - t, u(T - t), x(T - t)).transpose()) @
                    np.atleast_1d(y) + self.g.dx(u(T - t),
                                                 x(T - t)))
        p_sol = integrate.solve_ivp(func, (0, T), np.atleast_1d(self.h.dx(x(T))), dense_output=True, atol=1e-12,
                                    rtol=1e-12).sol

        def p_eval(t):
            return p_sol(T - t)

        p = TimeFunction(f=p_eval)
        if self.g == 0:
            def grad_eval(t):
                return np.atleast_1d(
                    np.atleast_1d(self.system.f.du(t, u(t), x(t)).transpose()) @
                    np.atleast_1d(p(t)))

        else:
            def grad_eval(t):
                return np.atleast_1d(
                    np.atleast_1d(self.system.f.du(t, u(t), x(t)).transpose()) @
                    np.atleast_1d(p(t))) + np.atleast_1d(
                    self.g.du(u(t), x(t)))

        print(plots)
        if plots is not None:
            solution = calculate_orbit(x(T))
            plots[0].set_data(solution[0], solution[1])
            plt.pause(0.01)
            plots[1].set_data(time_array_u, u.vector[::2])
            plt.pause(0.01)
            plots[2].set_data(time_array_u, u.vector[1::2])
            plt.pause(0.01)
        grad = TimeFunction(grad_eval)
        end = time.time()
        grad_time += end - start
        # print("Grad time: ", grad_time)
        return grad

    def grad_vector(self, u, plots=None):
        u.to_func()
        grad = self.grad(u, plots)
        grad.to_vector()
        grad.vector = T / n * grad.vector
        return grad

    def grad_wrapper(self, vector, plots=None):
        start = time.time()
        u = TimeFunction(vector=vector, dim=int(np.size(vector) / n))
        grad = self.grad_vector(u, plots)
        end = time.time()
        return grad.vector

    def J_wrapper(self, vector, plots=None):
        global J_time
        start = time.time()
        u = TimeFunction(vector=vector, dim=int(np.size(vector) / n))
        j = self(u)
        end = time.time()
        J_time = J_time + end - start
        # print("J time: ", J_time)
        return j


def calculate_orbit(x_at_t0):
    x = x_at_t0[0]
    y = x_at_t0[1]
    vx = x_at_t0[2]
    vy = x_at_t0[3]
    r = np.sqrt(x ** 2 + y ** 2)
    v = np.sqrt(vx ** 2 + vy ** 2)
    energy = v ** 2 / 2 - alpha / r
    theta1 = (vx * y - x * vy) / (x ** 2 + y ** 2)
    moment = r ** 2 * theta1
    ecc = np.sqrt(1 + 2 * energy * moment ** 2 / alpha ** 2)
    theta_array = np.linspace(0, 2 * np.pi, 1000)
    r_array = -moment ** 2 / alpha / (1 + ecc * np.cos(theta_array))
    x_array = r_array * np.cos(theta_array)
    y_array = r_array * np.sin(theta_array)
    return [x_array, y_array]
