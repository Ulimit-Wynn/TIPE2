import numpy as np
import scipy.integrate as integrate
import time


kg_to_mass_unit__coeff = 1 / 16290
meter_to_distance_unit_coeff = 1 / 637100
time_coff = 1 / 60
newton_to_force_unit_coeff = kg_to_mass_unit__coeff * meter_to_distance_unit_coeff / (time_coff ** 2)
T = 600 * time_coff
n = 50
dt = T / n
grad_time = 0
J_time = 0
to_vector_time = 0
integration_time = 0
solving_time = 0
Grav = 6.673 * (10 ** (-11)) * meter_to_distance_unit_coeff ** 3 / kg_to_mass_unit__coeff / (time_coff ** 2)
M = 5.972 * (10 ** 24) * kg_to_mass_unit__coeff
p0 = 101325 * kg_to_mass_unit__coeff / meter_to_distance_unit_coeff / (time_coff ** 2)
h0 = 8635 * meter_to_distance_unit_coeff
A = (0.25 * 0.75 ** 2) * np.pi * meter_to_distance_unit_coeff ** 2
Cd = 0.2
isp = 400 * time_coff
g0 = 9.81 * meter_to_distance_unit_coeff / (time_coff ** 2)
e0 = 0.25
a0 = 10 + 900000 * meter_to_distance_unit_coeff
v_ideal = 7400 * meter_to_distance_unit_coeff / time_coff
rocket_radius = 0.75 * meter_to_distance_unit_coeff
rocket_height = 16 * meter_to_distance_unit_coeff
rocket_inertia = 1/12 * (3 * rocket_radius ** 2 + rocket_height ** 2) + 1/4 * rocket_height ** 2


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
        #v = np.array([(self.integrate(i * period / step, (i + 1) * period / step) * step / period) for i in range(0, step)])
        v = np.array([1 / 2 * self(dt * (i + 1 / (2 * np.sqrt(3)))) + 1 / 2 *
                      self(dt * (i + 1 - 1 / (2 * np.sqrt(3)))) for i in range(0, step)])
        v = v.ravel()
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

    def solve_mass(self, u):
        def func(t, m_at_t):
            dm = np.array([-(u.evaluate(t)[0] + u.evaluate(t)[1]) / isp / g0])
            return dm

        solve = integrate.solve_ivp(func, (0, T), np.array([1]), dense_output=True, atol=1e-7, rtol=1e-7).sol
        return solve

    def solve_theta1(self, u, m):
        def func(t, theta1):
            theta2 = np.array([rocket_radius / (rocket_inertia * m(t)) * (u(t)[1] - u(t)[0])])
            return theta2

        solve = integrate.solve_ivp(func, (0, T), np.array([0]), dense_output=True, atol=1e-12, rtol=1e-12).sol
        return solve

    def solve_decoupled(self, u):
        start = time.time()
        m = self.solve_mass(u)
        theta1 = self.solve_theta1(u, m)

        def new_f(t, u_at_t, r_v_at_t, theta1_at_t, m_at_t):
            temp = np.array([theta1_at_t[0], m_at_t[0]])
            return self.f.evaluate(t, u_at_t, np.concatenate((r_v_at_t, temp)))[:-2]

        def func(t, r_v_at_t):
            drv = new_f(t, u(t), r_v_at_t, theta1(t), m(t))
            return drv
        solve = solve = integrate.solve_ivp(func, (0, T), self.x0[:-2], dense_output=True, atol=1e-12, rtol=1e-12).sol

        def solution_eval(t):
            temp = np.array([theta1(t)[0], m(t)[0]])
            return np.concatenate((solve(t), temp))

        solution = TimeFunction(f=solution_eval)
        print("decoupled time: ", time.time() - start)
        return solution

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

    def grad(self, u):
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
        p_sol = integrate.solve_ivp(func, (0, T), np.atleast_1d(self.h.dx(x(T))), dense_output=True, atol=1e-12, rtol=1e-12).sol

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

        grad = TimeFunction(grad_eval)
        end = time.time()
        grad_time += end - start
        #print("Grad time: ", grad_time)
        return grad

    def grad_vector(self, u):
        u.to_func()
        grad = self.grad(u)
        grad.to_vector()
        grad.vector = T / n * grad.vector
        return grad

    def grad_wrapper(self, vector):
        start = time.time()
        u = TimeFunction(vector=vector, dim=int(np.size(vector) / n))
        grad = self.grad_vector(u)
        end = time.time()
        return grad.vector

    def J_wrapper(self, vector):
        global J_time
        start = time.time()
        u = TimeFunction(vector=vector, dim=int(np.size(vector) / n))
        j = self(u)
        end = time.time()
        J_time = J_time + end - start
        # print("J time: ", J_time)
        return j
