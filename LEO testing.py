import scipy.optimize as optimize
from Solver import *
import numpy as np
import matplotlib.pyplot as plt
import sympy
import time as chrono

alpha = Grav * M
x0 = np.array([10, 0, 0, 0, 0, 0, 1])
t = sympy.symbols('t')
x, y, vx, vy, theta, theta1, m = sympy.symbols('x y vx vy theta theta1 m')
T1, T2 = sympy.symbols('T1 T2')
X = sympy.Matrix([x, y, vx, vy, theta, theta1, m])
U = sympy.Matrix([T1, T2])
r = sympy.sqrt(x ** 2 + y ** 2)
v = sympy.sqrt((vx ** 2 + vy ** 2))
r1 = (x * vx + y * vy) / r
thrust = T1 + T2
phi = sympy.atan2(y, x)
phi1 = (vx * y - x * vy) / (x ** 2 + y ** 2)
thrust_matrix = np.zeros((n, 2 * n))
fuel_matrix = np.zeros(2 * n)
inertia = rocket_inertia * m
drag = 0.5 * p0 * sympy.exp(-(r - 1) / h0) * Cd * A * v ** 2
ax = thrust * sympy.cos(theta) / m - alpha * x / (r ** 3) + vx * thrust / (isp * g0 * m) - drag * vx
ay = thrust * sympy.sin(theta) / m - alpha * y / (r ** 3) + vy * thrust / (isp * g0 * m) - drag * vy
mdot = -thrust / isp / g0
theta2 = (0.75 * meter_to_distance_unit_coeff / inertia) * (T2 - T1)
e_theta = np.array([1, y/x])
e_theta = 1/sympy.sqrt(1 + (y/x) ** 2) * e_theta
v_vector = np.array([vx, vy])
r_cross_v = vx * y - vy * x


F = sympy.Matrix([vx, vy, ax, ay, theta1, theta2, mdot])
H = (r_cross_v - v_ideal * a0) ** 2 / ((v_ideal * a0) ** 2) + (r - a0) ** 2 / (a0 ** 2) + (vx * x + vy * y) ** 2
G = 0 * (thrust / (isp * g0))
dFdU = F.jacobian(U)
dFdX = F.jacobian(X)
dHdX = H.diff(X)
dGdU = G.diff(U)
dGdX = G.diff(X)
f_tem = sympy.lambdify((t, U, X), F)
dfdx = sympy.lambdify((t, U, X), dFdX)
dfdu_tem = sympy.lambdify((t, U, X), dFdU)
h_eval = sympy.lambdify((X,), H)
dhdx_tem = sympy.lambdify((X,), dHdX)
g_eval = sympy.lambdify((U, X), G)
dgdx_tem = sympy.lambdify((U, X), dGdX)
dgdu_tem = sympy.lambdify((U, X), dGdU)


def f_eval(time_value, u_at_t, x_at_t):
    return f_tem(time_value, u_at_t, x_at_t).ravel()


def dfdu(time_value, u_at_t, x_at_t):
    return dfdu_tem(time_value, u_at_t, x_at_t)


def dgdu(u_at_t, x_at_t):
    return dgdu_tem(u_at_t, x_at_t).ravel()


def dgdx(u_at_t, x_at_t):
    return dgdx_tem(u_at_t, x_at_t).ravel()


def dhdx(x_at_t):
    return dhdx_tem(x_at_t).ravel()


def u_eval(time_value):
    if time_value < 70 * time_coff:
        return np.array([9.806 * 6000 * newton_to_force_unit_coeff, 9.806 * 6000 * newton_to_force_unit_coeff])
    else:
        return np.array([9.806 * (7000 - 1000 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff, 9.806 * (7000 - 1000.1 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff])


def thrust_constraint_min(vector):
    return vector


def thrust_constraint_max(vector):
    res = 16000 * 9.806 * newton_to_force_unit_coeff * np.ones(np.size(vector)) - vector
    return res


def fuel_constraint(vector):
    return (16290 - 550) * kg_to_mass_unit__coeff - np.sum(vector) * dt / isp / g0


def solve_for_orbit(x_at_t0):
    def func(time_value, y):
        dy = np.array([y[2], y[3], - alpha * y[0] / (np.sqrt(y[0] ** 2 + y[1] ** 2) ** 3),
                       - alpha * y[1] / (np.sqrt(y[0] ** 2 + y[1] ** 2) ** 3), 0])
        return dy
    simple_x_at_0 = np.array([x_at_t0[0], x_at_t0[1], x_at_t0[2], x_at_t0[3], x_at_t0[6]])
    solve = integrate.solve_ivp(func, (0, 600), simple_x_at_0, dense_output=True, rtol=1e-13, atol=1e-8).sol
    solution = TimeFunction(f=solve.__call__)
    return solution


f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=dhdx)
system = DynamicalSystem(f, x0)
J = Functional(system, g, h)
u = TimeFunction(u_eval)
u.to_vector()
print(system.solve_decoupled(u)(T))
solution = system.solve(u)
"""d = np.ones(np.size(grad.vector))
for i in range(3, 10):
    print("i = ", i)
    print("")
    eps = 10 ** (-i)
    u_test1 = TimeFunction(vector=u.vector + eps * d, dim=u.dim)
    u_test2 = TimeFunction(vector=u.vector - eps * d, dim=u.dim)
    print("J +: ", J(u_test1))
    print("J -: ", J(u_test2))
    print("d @ grad: ", d @ grad.vector)
    print("Finite difference: ", (J(u_test1) - J(u_test2)) / (2 * eps))
    print("")
"""
drag_dX = drag.diff(X)
print(drag_dX[2])
drag_func = sympy.lambdify((X,), drag)
drag_dX_func = sympy.lambdify((X,), drag_dX)
print("drag: ", f.dx(0, u(0), np.array([10, 0, 0, 0, 0, 0, 1])))

