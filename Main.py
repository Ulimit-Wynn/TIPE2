import scipy.optimize as optimize
from Solver import *
import numpy as np
import matplotlib.pyplot as plt
import sympy
import time as chrono

alpha = Grav * M
x0 = np.array([10, 0, 0, 0, 1])
t = sympy.symbols('t')
x, y, vx, vy, m = sympy.symbols('x y vx vy m')
thrust, theta = sympy.symbols('thrust theta')
X = sympy.Matrix([x, y, vx, vy, m])
U = sympy.Matrix([thrust, theta])
r = sympy.sqrt(x ** 2 + y ** 2)
v = sympy.sqrt((vx ** 2 + vy ** 2))
r1 = (x * vx + y * vy) / r
phi = sympy.atan2(y, x)
phi1 = (vx * y - x * vy) / (x ** 2 + y ** 2)
thrust_matrix = np.zeros((n, 2 * n))
fuel_matrix = np.zeros(2 * n)
for i in range(0, n):
    thrust_matrix[i][2 * i] = 1
for i in range(0, n):
    fuel_matrix[2 * i] = dt / isp
drag = 0.5 * p0 * sympy.exp(-(r - 1) / h0) * Cd * A * v
ax = thrust * sympy.cos(theta) / m - alpha * x / (r ** 3) + vx * thrust / (isp * g0 * m) - drag * vx
ay = thrust * sympy.sin(theta) / m - alpha * y / (r ** 3) + vy * thrust / (isp * g0 * m) - drag * vy
mdot = -thrust / isp / g0

energy = (1 / 2 * (vx ** 2 + vy ** 2) - alpha / r)
moment = (vx * y - x * vy)

F = sympy.Matrix([vx, vy, ax, ay, mdot])
H = (v - v_ideal) ** 2 / (v_ideal ** 2) + (r - a0) ** 2 / (a0 ** 2) + (vx * x + vy * y) ** 2
G = 0 * (thrust / (isp * g0)) + 0.001 / (r - 9.2)
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
    if u_at_t[0] == 0:
        return np.array([[0, 0], [0, 0], [1 / x_at_t[4], 0], [0, 1 / x_at_t[4]], [0, 0]])
    return dfdu_tem(time_value, u_at_t, x_at_t)


def dgdu(u_at_t, x_at_t):
    return dgdu_tem(u_at_t, x_at_t).ravel()


def dgdx(u_at_t, x_at_t):
    return dgdx_tem(u_at_t, x_at_t).ravel()


def dhdx(x_at_t):
    return dhdx_tem(x_at_t).ravel()


def u_eval(time_value):
    if time_value < 70 * time_coff:
        return np.array([9.806 * 21920 * newton_to_force_unit_coeff, 0])
    else:
        return np.array([9.806 * 11920 * newton_to_force_unit_coeff, (time_value - 70 * time_coff) * np.pi/T])


def thrust_constraint(vector):
    res = thrust_matrix @ vector
    return res


def fuel_constraint(vector):
    return (16290 - 550) * kg_to_mass_unit__coeff - np.sum(vector[::2]) * dt / isp / g0



def solve_for_orbit(x_at_t0):
    def func(time_value, y):
        dy = np.array([y[2], y[3], - alpha * y[0] / (np.sqrt(y[0] ** 2 + y[1] ** 2) ** 3),
                       - alpha * y[1] / (np.sqrt(y[0] ** 2 + y[1] ** 2) ** 3), 0])
        return dy

    solve = integrate.solve_ivp(func, (0, 600), x_at_t0, dense_output=True, rtol=10 ** (-13), atol=10 ** (-8)).sol
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

result = optimize.minimize(J.J_wrapper, u.vector, tol=1e-4, jac=J.grad_wrapper,
                           constraints=({"type": "ineq", "fun": thrust_constraint},
                                        {"type": "ineq", "fun": fuel_constraint}))
print(result)
u1 = TimeFunction(vector=result.x, dim=2)
u1.to_func()
grad = result.jac
d = np.ones(np.size(u.vector))
for i in range(2, 10):
    print(" ")
    print("eps=", "10^-", str(i))
    eps = 10 ** (-1 * i)
    u_test1 = TimeFunction(vector=u1.vector + eps * d, dim=u.dim)
    u_test2 = TimeFunction(vector=u1.vector - eps * d, dim=u.dim)
    print("d @ grad: ", d @ grad)
    print("Finite difference: ", (J(u_test1) - J(u_test2)) / 2 / eps)
time_array_u = np.linspace(0, T, n)
time_array_x = np.linspace(0, T, 20000)
P_original = system.solve(u)
P = TimeFunction(f=system.solve(u1))
P_original.to_vector(step=20000)
P.to_vector(step=20000)
orbit_original = solve_for_orbit(P_original(T))
orbit = solve_for_orbit(P(T))
orbit_original.to_vector(step=20000, period=600)
orbit.to_vector(step=20000, period=600)
orbit_x = orbit.vector[::5]
orbit_y = orbit.vector[1::5]
orbit_original_x = orbit_original.vector[::5]
orbit_original_y = orbit_original.vector[1::5]
earth_x = 10 * np.array([np.cos(i * 2 * np.pi / 20000) for i in range(0, 20000)])
earth_y = 10 * np.array([np.sin(i * 2 * np.pi / 20000) for i in range(0, 20000)])
ideal_orbit_x = a0 * np.array([np.cos(i * 2 * np.pi / 20000) for i in range(0, 20000)])
ideal_orbit_y = a0 * np.array([np.sin(i * 2 * np.pi / 20000) for i in range(0, 20000)])
print(P(T))
X1 = P.vector[::5]
X2 = P.vector[1::5]
Mass = P.vector[4::5]
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
plt.plot(orbit_original_x, orbit_original_y)
plt.plot(ideal_orbit_x, ideal_orbit_y)
end = chrono.time()
print("time: ", end - start)
plt.show()
