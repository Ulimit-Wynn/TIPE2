import scipy.optimize as optimize
import time as chrono
from functions import *
import numpy as np


_3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia = .75 * meter_to_distance_unit_coeff / rocket_inertia


def u_eval(time_value):
    if time_value < 70 * time_coff:
        return np.array([9.806 * 3500 * newton_to_force_unit_coeff, 9.806 * 3500 * newton_to_force_unit_coeff])
    else:
        return np.array([9.806 * (3500 - 500 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff,
                         9.806 * (3500 - 500.1 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff])


def positive_fuel(vector):
    return vector[0] - 550 * kg_to_mass_unit__coeff


def thrust_constraint_min_variable_mass(vector):
    return vector[1::]


def thrust_constraint_max_variable_mass(vector):
    res = 16000 * 9.806 * newton_to_force_unit_coeff * np.ones(np.size(vector[1::])) - vector[1::]
    return res


def fuel_constraint_variable_mass(vector):
    return vector[0] - 550 * kg_to_mass_unit__coeff - np.sum(vector[1::]) * dt / isp / g0


def thrust_constraint_min(vector):
    return vector[::]


def thrust_constraint_max(vector):
    res = 16000 * 9.806 * newton_to_force_unit_coeff * np.ones(np.size(vector[::])) - vector[::]
    return res


def fuel_constraint(vector):
    return initial_control[0] - 550 * kg_to_mass_unit__coeff - np.sum(vector[::]) * dt / isp / g0


def solve_for_orbit(x_at_t0):
    def func(time_value, y):
        dy = np.array([y[2], y[3], - alpha * y[0] / (np.sqrt(y[0] ** 2 + y[1] ** 2) ** 3),
                       - alpha * y[1] / (np.sqrt(y[0] ** 2 + y[1] ** 2) ** 3), 0])
        return dy

    simple_x_at_0 = np.array([x_at_t0[0], x_at_t0[1], x_at_t0[2], x_at_t0[3], x_at_t0[6]])
    solve = integrate.solve_ivp(func, (0, 600), simple_x_at_0, dense_output=True, rtol=1e-13, atol=1e-8).sol
    solution = TimeFunction(f=solve.__call__)
    print("orbit solved")
    return solution


initial_control = np.load('2019_05_02__20_14_29__consecutive_diff_45%_extra_mass_T600_n200.npy')
system.x0[6] = initial_control[0] * (1 + 4/10)
u = TimeFunction(vector=initial_control[1::], dim=2)
# u.to_func(step=T/100)
# u.to_vector()
# u.vector = np.concatenate((np.array([system.x0[6]]), u.vector))

def h_constraint(vector, x_at_t=None):
    if not (x_at_t is None):
        return np.array([h_r.evaluate(x_at_t), h_v.evaluate(x_at_t), h_dot.evaluate(x_at_t), h_theta1.evaluate(x_at_t)])
    u = TimeFunction(vector=vector, dim=2)
    u.to_func()
    x = system.solve(u)
    constraint = np.array([h_r.evaluate(x(T)), h_v.evaluate(x(T)), h_dot.evaluate(x(T)), h_theta1.evaluate(x(T))])
    print("h: ", constraint)
    return constraint


def h_constraint_grad(vector):
    return np.stack((J_r.grad_wrapper(vector),
                     J_v.grad_wrapper(vector),
                     J_dot.grad_wrapper(vector),
                     J_theta1.grad_wrapper(vector)))


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
    solution = calculate_orbit(x_T)
    orbit_ani.set_data(solution[0], solution[1])
    plt.pause(0.01)
    control1.set_data(time_array_u, u.vector[::2])
    plt.pause(0.01)
    control2.set_data(time_array_u, u.vector[1::2])
    plt.pause(0.01)
    return T / n * np.concatenate(grad_list, axis=1)


def h_mass_wrapper(vector):
    system.x0[6] = vector[0]
    return h_constraint(vector[1::])


def better_h_mass_grad(vector):
    system.x0[6] = vector[0]
    v = vector[1::]
    u = TimeFunction(vector=v, dim=2)
    u.to_func()
    x = system.solve(u)
    x_t = x(T)

    def func(t, y):
        matrix_y = np.reshape(y, (4, 7))
        return np.ravel(matrix_y @ dfdx(T - t, u(T - t), x(T - t)))

    def func_m(t, y):
        return f.dx(t, u(t), x(t)) @ y

    p_sol = integrate.solve_ivp(func, (0, T), np.concatenate((dhdx_r(x_t), dhdx_v(x_t), dhdx_dot(x_t), dhdx_theta1(x_t))), dense_output=True, atol=1e-12, rtol=1e-12).sol
    dx = integrate.solve_ivp(func_m, (0, T), np.array([0, 0, 0, 0, 0, 0, 1]), dense_output=True, atol=1e-12, rtol=1e-12).sol

    def p_eval(t):
        return np.reshape(p_sol(T - t), (4, 7))

    def grad_eval(t):
        return p_eval(t) @ dfdu(t, u(t), x(t))
    grad_list = [sum([W_i[j] * grad_eval(dt * (i + X_i[j])) for j in range(deg)]) for i in range(0, n)]
    grad_m = dhdm(x(T), dx(T))

    solution = calculate_orbit(x_t)
    orbit_ani.set_data(solution[0], solution[1])
    plt.pause(0.01)
    control1.set_data(time_array_u, u.vector[::2])
    plt.pause(0.01)
    control2.set_data(time_array_u, u.vector[1::2])
    plt.pause(0.01)
    grad_thrust = T / n * np.concatenate(grad_list, axis=1)
    results = np.column_stack((grad_m, grad_thrust))
    return results/2


def h_dot_inequality_max(vector):
    system.x0[6] = vector[0]
    return 0.3 - J_dot.J_wrapper(vector[1::])


def h_dot_inequality_min(vector):
    system.x0[6] = vector[0]
    return J_dot.J_wrapper(vector[1::]) + 0.3


def consecutive_diff(vector, plots=None):
    return sum((vector[i] - vector[i-2])**2 for i in range(2, len(vector)))


def consecutive_diff_grad(vector, plots=None):
    size = len(vector)
    return np.concatenate((np.array([-2 * (vector[2] - vector[0]), -2 * (vector[3] - vector[1])]),
                           np.array([4 * vector[i] - 2 * (vector[i+2] + vector[i-2]) for i in range(2, size-2)]),
                           np.array([2 * (vector[size-2] - vector[size-4]), 2 * (vector[size-1] - vector[size-3])])))


def initial_mass(vector):
    return vector[0]


def initial_mass_grad(vector):
    grad = np.zeros(np.size(vector))
    grad[0] = 1
    return grad

start = chrono.time()
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(111)
ax21 = fig2.add_subplot(211)
ax22 = fig2.add_subplot(212)
ax1.set_xlim(-30,30)
ax1.set_ylim(-30,30)
ax21.set_xlim(0, T)
ax21.set_ylim(0, 16000 * 9.806 * newton_to_force_unit_coeff)
ax22.set_xlim(0, T)
ax22.set_ylim(0, 16000 * 9.806 * newton_to_force_unit_coeff)
plt.ion()
plt.show()
ax1.plot(ideal_orbit_x, ideal_orbit_y)
orbit_ani, = ax1.plot([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])
control1, = ax21.step([0], [0])
control2, = ax22.step([0], [0])
plt.pause(1)

result = optimize.minimize(consecutive_diff, u.vector, method="SLSQP", jac=consecutive_diff_grad,options={"maxiter": 1000, "ftol": 1e-6, "iprint": 3, "disp": True},
                           constraints=({"type": "ineq", "fun": thrust_constraint_min},
                                        {"type": "ineq", 'fun': thrust_constraint_max},
                                        {"type": "ineq", "fun": fuel_constraint},
                                        {"type": "eq", "fun": h_constraint, "jac": better_h_constraint_grad}))
"""
result = optimize.minimize(initial_mass, u.vector, method="SLSQP", jac=initial_mass_grad, options={"maxiter": 10000, 'ftol': 1e-8,"iprint": 3, "disp": True},
                           constraints=({"type": "ineq", "fun": thrust_constraint_min_variable_mass},
                                        {"type": "ineq", 'fun': thrust_constraint_max_variable_mass},
                                        {"type": "ineq", "fun": fuel_constraint_variable_mass},
                                        {"type": "ineq", "fun": positive_fuel},
                                        {"type": "eq", "fun": h_mass_wrapper, "jac": better_h_mass_grad}))
"""
print(result)
np.save(time.strftime("%Y_%m_%d__%H_%M_%S__") + "consecutive_diff_40%_extra_mass_T600_n200.npy", np.concatenate((np.array([system.x0[6]]), result.x)))
end = chrono.time()
print("total time taken: ", end - start)
