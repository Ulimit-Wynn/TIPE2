import scipy.optimize as optimize
import time as chrono
from Functions import *
import numpy as np


_3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia = .75 * meter_to_distance_unit_coeff / rocket_inertia


def u_eval(time_value):
    if time_value < 70 * time_coff:
        return np.array([9.806 * 3500 * newton_to_force_unit_coeff, 9.806 * 3500 * newton_to_force_unit_coeff])
    else:
        return np.array([9.806 * (3500 - 500 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff,
                         9.806 * (3500 - 500.1 * (time_value - 70 * time_coff)) * newton_to_force_unit_coeff])


def positive_fuel(vector):
    return vector[0]


def thrust_constraint_min(vector):
    return vector[1::]


def thrust_constraint_max(vector):
    res = 16000 * 9.806 * newton_to_force_unit_coeff * np.ones(np.size(vector[1::])) - vector[1::]
    return res


def fuel_constraint(vector):
    return vector[0] - 550 * kg_to_mass_unit__coeff - np.sum(vector[1::]) * dt / isp / g0


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
        P = np.reshape(y, (4, 7))
        return np.ravel(P @ dfdx(T - t, u(T - t), x(T - t)))
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


def h_mass_grad(vector):
    system[6] = vector[0]
    return better_h_constraint_grad(vector[1::])


def consecutive_diff(vector, plots=None):
    return sum((vector[i] - vector[i-2])**2 for i in range(2, len(vector)))


def initial_mass(vector):
    return vector[0]


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

result = optimize.minimize(consecutive_diff, u.vector, [orbit_ani, control1, control2], method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000, "iprint": 3, "disp": True},
                           constraints=({"type": "ineq", "fun": thrust_constraint_min},
                                        {"type": "ineq", 'fun': thrust_constraint_max},
                                        {"type": "ineq", "fun": fuel_constraint},
                                        {"type": "eq", "fun": h_constraint, "jac": better_h_constraint_grad}))
"""result = optimize.minimize(J_h.J_wrapper, u.vector, [orbit_ani, control1, control2], method="SLSQP", options={"ftol": 1e-15, "maxiter": 100, "iprint": 3, "disp": True}, jac=J_h.grad_wrapper,
                           constraints=({"type": "ineq", "fun": thrust_constraint_min},
                                        {"type": "ineq", 'fun': thrust_constraint_max},
                                        {"type": "ineq", "fun": fuel_constraint}))
"""
print(result)
u1 = TimeFunction(vector=result.x, dim=2)
np.save("consecutive_diff_T600_n50", u1.vector)
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
plt.ylim((15, -15))
plt.xlim((15, -15))
plt.autoscale(False)
plt.plot(earth_x, earth_y)
plt.plot(X1, X2)
plt.figure(2)
plt.subplot(211)
plt.step(time_array_u, u1.vector[::2])
plt.step(time_array_u, u.vector[::2])
plt.subplot(212)
plt.step(time_array_u, u1.vector[1::2])
plt.step(time_array_u, u.vector[1::2])
plt.figure(3)
plt.plot(earth_x, earth_y)
plt.plot(orbit_x, orbit_y)
plt.plot(ideal_orbit_x, ideal_orbit_y)
end = chrono.time()
print("time: ", end - start)
plt.show()
