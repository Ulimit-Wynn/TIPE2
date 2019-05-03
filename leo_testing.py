import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import sympy
from functions import *

alpha = Grav * M
x0 = np.array([10., 0., 0., 0., 0., 0., 1.])
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
ax = thrust * sympy.cos(theta) / m - alpha * x / (r ** 3) + vx * thrust / (isp * g0 * m) - drag * vx
ay = thrust * sympy.sin(theta) / m - alpha * y / (r ** 3) + vy * thrust / (isp * g0 * m) - drag * vy
m_dot = -thrust / isp / g0
theta2 = (0.75 * meter_to_distance_unit_coeff / inertia) * (T2 - T1)
e_theta = np.array([1, y / x])
e_theta = 1 / sympy.sqrt(1 + (y / x) ** 2) * e_theta
v_vector = np.array([vx, vy])
r_cross_v = vx * y - vy * x


def u_eval(time_value):
    if time_value < 70 * time_coff:
        return np.array([9.806 * 3500 * newton_to_force_unit_coeff, 9.806 * 3500 * newton_to_force_unit_coeff])
    else:
        return np.array([9.806 * (3500 - 500 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff,
                         9.806 * (3500 - 500.1 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff])


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
h_zero = DifferentiableFunction(f=h_eval_zero, dfdx=dhdx_zero)
h = DifferentiableFunction(f=h_eval, dfdx=dhdx)
h_r = DifferentiableFunction(f=h_eval_r, dfdx=dhdx_r)
h_v = DifferentiableFunction(f=h_eval_v, dfdx=dhdx_v)
h_dot = DifferentiableFunction(f=h_eval_dot, dfdx=dhdx_dot)
h_theta1 = DifferentiableFunction(f=h_eval_theta1, dfdx=dhdx_theta1)
system = DynamicalSystem(f, x0)
J = Functional(system, g, h)
J_zero = Functional(system, g, h_zero)
J_h = Functional(system, 0, h)
J_r = Functional(system, 0, h_r)
J_v = Functional(system, 0, h_v)
J_dot = Functional(system, 0, h_dot)
J_theta1 = Functional(system, 0, h_theta1)
u = TimeFunction(f=u_eval)
u.vector = (np.load("least_square.npy"))


# u.to_vector()

def h_constraint(vector, x_at_t=None):
    if not (x_at_t is None):
        return np.array([h_r.evaluate(x_at_t), h_v.evaluate(x_at_t), h_dot.evaluate(x_at_t), h_theta1.evaluate(x_at_t)])
    u = TimeFunction(vector=vector, dim=2)
    u.to_func()
    x = system.solve(u)
    constraint = np.array([h_r.evaluate(x(T)), h_v.evaluate(x(T)), h_dot.evaluate(x(T)), h_theta1.evaluate(x(T))])
    print("h: ", constraint)
    return constraint


def h_mass_wrapper(vector):
    system.x0[6] = vector[0]
    return h_constraint(vector[1::])


def better_h_mass_grad(vector):
    system.x0[6] = vector[0]
    v = vector[1::]
    u = TimeFunction(vector=v, dim=2)
    u.to_func()
    x = system.solve(u)
    x_T = x(T)

    def func(t, y):
        P = np.reshape(y, (4, 7))
        return np.ravel(P @ dfdx(T - t, u(T - t), x(T - t)))

    def func_m(t, y):
        return f.dx(t, u(t), x(t)) @ y

    p_sol = integrate.solve_ivp(func, (0, T),
                                np.concatenate((dhdx_r(x_T), dhdx_v(x_T), dhdx_dot(x_T), dhdx_theta1(x_T))),
                                dense_output=True, atol=1e-12, rtol=1e-12).sol
    dx = integrate.solve_ivp(func_m, (0, T), np.array([0, 0, 0, 0, 0, 0, 1]), dense_output=True, atol=1e-12,
                             rtol=1e-12).sol

    def p_eval(t):
        return np.reshape(p_sol(T - t), (4, 7))

    def grad_eval(t):
        return p_eval(t) @ dfdu(t, u(t), x(t))

    grad_list = [sum([W_i[j] * grad_eval(dt * (i + X_i[j])) for j in range(deg)]) for i in range(0, n)]
    grad_m = dhdm(x(T), dx(T))

    grad_thrust = T / n * np.concatenate(grad_list, axis=1)
    results = np.column_stack((grad_m, grad_thrust))
    return results


def new_better_h_constraint_grad(vector):
    u = TimeFunction(vector=vector, dim=2)
    u.to_func()
    x = system.solve(u)
    x_T = x(T)

    def func(t, y):
        y_matrix = np.reshape(y[:28], (4, 7))
        dx = -f.evaluate(T - t, u.evaluate(T - t), y[28:])
        return np.concatenate((np.ravel(y_matrix @ dfdx(T - t, u(T - t), y[28:])), dx))

    p_sol = integrate.solve_ivp(func, (0, T), np.concatenate((dhdx_r(x_T), dhdx_v(x_T), dhdx_dot(x_T), dhdx_theta1(x_T), x_T)), dense_output=True, atol=1e-12, rtol=1e-12).sol

    def p_eval(t):
        return np.reshape(p_sol(T - t)[:28], (4, 7))

    def grad_eval(t):
        return p_eval(t) @ dfdu(t, u(t), x(t))
    grad_list = [sum([W_i[j] * grad_eval(dt * (i + X_i[j])) for j in range(deg)]) for i in range(0, n)]
    return T / n * np.concatenate(grad_list, axis=1)


def better_h_constraint_grad(vector):
    u = TimeFunction(vector=vector, dim=2)
    u.to_func()
    x = system.solve(u)
    x_T = x(T)

    def func(t, y):
        y_matrix = np.reshape(y, (4, 7))
        return np.ravel(y_matrix @ dfdx(T - t, u(T - t), x(T - t)))

    p_sol = integrate.solve_ivp(func, (0, T), np.concatenate((dhdx_r(x_T), dhdx_v(x_T), dhdx_dot(x_T), dhdx_theta1(x_T))), dense_output=True, atol=1e-12, rtol=1e-12).sol

    def p_eval(t):
        return np.reshape(p_sol(T - t), (4, 7))

    def grad_eval(t):
        return p_eval(t) @ dfdu(t, u(t), x(t))
    grad_list = [sum([W_i[j] * grad_eval(dt * (i + X_i[j])) for j in range(deg)]) for i in range(0, n)]
    return T / n * np.concatenate(grad_list, axis=1)

vector = np.load("variable_mass_T600_n50.npy")[1::]
print((new_better_h_constraint_grad(vector) - better_h_constraint_grad(vector)) / better_h_constraint_grad(vector))


"""vector = np.load("variable_mass_T600_n50.npy")
print(vector[0])
u = TimeFunction(vector=vector[1:], dim=2)
u.to_func()
system.x0[6] = vector[0]
x_final = system.solve(u)
data = calculate_orbit(x_final(T))
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(111)
ax21 = fig2.add_subplot(211)
ax22 = fig2.add_subplot(212)
ax1.set_xlim(-30, 30)
ax1.set_ylim(-30, 30)
ax21.set_xlim(0, T)
ax21.set_ylim(0, 16000 * 9.806 * newton_to_force_unit_coeff)
ax22.set_xlim(0, T)
ax22.set_ylim(0, 16000 * 9.806 * newton_to_force_unit_coeff)
ax1.plot(ideal_orbit_x, ideal_orbit_y)
ax1.plot(data[0], data[1])
plt.show()"""