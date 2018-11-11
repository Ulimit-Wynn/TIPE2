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
inertia = rocket_inertia * m
drag = 0.5 * p0 * sympy.exp(-(r - 1) / h0) * Cd * A * v
ax = thrust * sympy.cos(theta) / m - alpha * x / (r ** 3) + vx * thrust / (isp * g0 * m) - 0 * drag * vx
ay = thrust * sympy.sin(theta) / m - alpha * y / (r ** 3) + vy * thrust / (isp * g0 * m) - 0 * drag * vy
m_dot = -thrust / isp / g0
theta2 = (0.75 * meter_to_distance_unit_coeff / inertia) * (T2 - T1)
e_theta = np.array([1, y/x])
e_theta = 1/sympy.sqrt(1 + (y/x) ** 2) * e_theta
v_vector = np.array([vx, vy])
r_cross_v = vx * y - vy * x


F = sympy.Matrix([vx, vy, ax, ay, theta1, theta2, m_dot])
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
        return np.array([9.806 * 7000 * newton_to_force_unit_coeff, 9.806 * 7000 * newton_to_force_unit_coeff])
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

start = chrono.time()

result = optimize.minimize(J.J_wrapper, u.vector, method="SLSQP", options={"ftol": 1e-12}, jac=J.grad_wrapper,
                           constraints=({"type": "ineq", "fun": thrust_constraint_min},
                                        {"type": "ineq", 'fun': thrust_constraint_max},
                                        {"type": "ineq", "fun": fuel_constraint}))
print(result)
u1 = TimeFunction(vector=result.x, dim=2)
u1.to_func()
grad = result.jac

time_array_u = np.linspace(0, T, n)
time_array_x = np.linspace(0, T, 5000)
P_original = system.solve(u)
P = TimeFunction(f=system.solve(u1))
P_original.to_vector(step=5000)
P.to_vector(step=5000)
orbit_original = solve_for_orbit(P_original(T))
orbit = solve_for_orbit(P(T))
orbit_original.to_vector(step=5000, period=600)
orbit.to_vector(step=5000, period=600)
orbit_x = orbit.vector[::5]
orbit_y = orbit.vector[1::5]
orbit_original_x = orbit_original.vector[::5]
orbit_original_y = orbit_original.vector[1::5]
earth_x = 10 * np.array([np.cos(i * 2 * np.pi / 5000) for i in range(0, 5000)])
earth_y = 10 * np.array([np.sin(i * 2 * np.pi / 5000) for i in range(0, 5000)])
ideal_orbit_x = a0 * np.array([np.cos(i * 2 * np.pi / 5000) for i in range(0, 5000)])
ideal_orbit_y = a0 * np.array([np.sin(i * 2 * np.pi / 5000) for i in range(0, 5000)])
print(P(T))
X1 = P.vector[::7]
X2 = P.vector[1::7]
Mass = P.vector[4::7]
gr1 = grad[::2]
gr2 = grad[1::2]
plt.figure(1)
plt.subplot(211)
plt.plot(time_array_x, X1)
plt.subplot(212)
plt.plot(time_array_x, X2)
plt.figure(2)
plt.ylim((15, -15))
plt.xlim((15, -15))
plt.autoscale(False)
plt.plot(earth_x, earth_y)
plt.plot(X1, X2)
plt.figure(3)
plt.subplot(211)
plt.plot(time_array_u, u1.vector[::2])
plt.plot(time_array_u, u.vector[::2])
plt.subplot(212)
plt.plot(time_array_u, u1.vector[1::2])
plt.plot(time_array_u, u.vector[1::2])
plt.figure(4)
plt.plot(earth_x, earth_y)
plt.plot(orbit_x, orbit_y)
plt.plot(ideal_orbit_x, ideal_orbit_y)
end = chrono.time()
print("time: ", end - start)
plt.show()
