import scipy.optimize as optimize
from Solver import *
import numpy as np
import matplotlib.pyplot as plt
import sympy
import time as chrono
import math, cmath

inv_isp_g0 = 1 / (isp * g0)
half_A_Cd_p0 = .5 * A * Cd * p0
_3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia = \
    .75 * meter_to_distance_unit_coeff / rocket_inertia
inv_h0 = 1. / h0
alpha = Grav * M
x0 = np.array([10, 0, 0, 0, 0, 0, 1])
t = sympy.symbols('t')
x, y, vx, vy, theta, theta1, m = sympy.symbols('x y vx vy theta theta1 m')
T1, T2 = sympy.symbols('T1 T2')
X = sympy.Matrix([x, y, vx, vy, theta, theta1, m])
U = sympy.Matrix([T1, T2])
r = sympy.sqrt(x ** 2 + y ** 2)
v = sympy.sqrt((vx ** 2 + vy ** 2))
thrust = T1 + T2
H_zero = 0 * x
H = (r - a0) ** 2 / a0 ** 2 + (v - v_ideal) ** 2 / v_ideal ** 2 + (vx * x + vy * y) ** 2 + theta1 ** 2
H_r = (r - a0) / a0
H_v = (v - v_ideal) / v_ideal
H_dot = (vx * x + vy * y)
H_theta1 = theta1
G = 0*(thrust / (isp * g0))
dHdX_zero = H_zero.diff(X)
dHdX = H.diff(X)
dHdX_r = H_r.diff(X)
dHdX_v = H_v.diff(X)
dHdX_dot = H_dot.diff(X)
dHdX_theta1 = H_theta1.diff(X)
dGdU = G.diff(U)
dGdX = G.diff(X)
h_eval_zero = sympy.lambdify((X,), H_zero)
h_eval = sympy.lambdify((X,), H)
h_eval_r = sympy.lambdify((X,), H_r)
h_eval_v = sympy.lambdify((X,), H_v)
h_eval_dot = sympy.lambdify((X,), H_dot)
h_eval_theta1 = sympy.lambdify((X,), H_theta1)
dhdx_tem_zero = sympy.lambdify((X,), dHdX_zero)
dhdx_tem = sympy.lambdify((X,), dHdX)
dhdx_tem_r = sympy.lambdify((X,), dHdX_r)
dhdx_tem_v = sympy.lambdify((X,), dHdX_v)
dhdx_tem_dot = sympy.lambdify((X,), dHdX_dot)
dhdx_tem_theta1 = sympy.lambdify((X,), dHdX_theta1)
g_eval = sympy.lambdify((U, X), G)
dgdx_tem = sympy.lambdify((U, X), dGdX)
dgdu_tem = sympy.lambdify((U, X), dGdU)


def f_eval(time_value, u_at_t, x_at_t):
    thrust = u_at_t[0] + u_at_t[1]
    x = x_at_t[0]
    y = x_at_t[1]
    vx = x_at_t[2]
    vy = x_at_t[3]
    r = np.sqrt(x**2 + y**2)
    v = np.sqrt(vx**2 + vy**2)
    m_inv = 1/x_at_t[6]
    drag = half_A_Cd_p0 * v * np.exp(-(r - 1) / h0)
    r_cubed = r * r * r
    ax = thrust * math.cos(x_at_t[4]) * m_inv - alpha * x / r_cubed + thrust * vx * inv_isp_g0 * m_inv - drag * vx
    ay = thrust * math.sin(x_at_t[4]) * m_inv - alpha * y / r_cubed + thrust * vy * inv_isp_g0 * m_inv - drag * vy
    theta2 = (_3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia * m_inv) * (u_at_t[1] - u_at_t[0])
    results = np.zeros(7)
    results[0] = vx
    results[1] = vy
    results[2] = ax
    results[3] = ay
    results[4] = x_at_t[5]
    results[5] = theta2
    results[6] = -thrust * inv_isp_g0
    return results


def dfdu(time_value_,u_at_t, x_at_t):
    vx = x_at_t[2]
    vy = x_at_t[3]
    m_inv = 1/x_at_t[6]
    theta = x_at_t[4]
    results = np.zeros((7,2))
    results[2, 0] = m_inv * (math.cos(theta) + vx * inv_isp_g0)
    results[2, 1] = m_inv * (math.cos(theta) + vx * inv_isp_g0)
    results[3, 0] = m_inv * (math.sin(theta) + vy * inv_isp_g0)
    results[3, 1] = m_inv * (math.sin(theta) + vy * inv_isp_g0)
    results[5, 0] = -_3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia * m_inv
    results[5, 1] = _3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia * m_inv
    results[6, 0] = -inv_isp_g0
    results[6, 1] = -inv_isp_g0
    return results


def dfdx(time_value, u_at_t, x_at_t):
    x = x_at_t[0]
    y = x_at_t[1]
    vx = x_at_t[2]
    vy = x_at_t[3]
    r = math.sqrt(x * x + y * y)
    v = math.sqrt(vx * vx + vy * vy)
    D = half_A_Cd_p0 * math.exp((1. - r) * inv_h0)

    result = np.zeros((7, 7))

    result[0, 2] = 1.
    result[1, 3] = 1.

    cos_sin = cmath.exp(complex(0., x_at_t[4]))
    alpha_inv_r5 = alpha * r ** -5
    _3_x_y_alpha_inv_r5 = 3. * x * y * alpha * math.pow(r, -5)

    m_inv = 1. / x_at_t[6]
    m_inv_thrust = m_inv * (u_at_t[0] + u_at_t[1])

    result[2, 0] = -(r * r - 3. * x * x) * alpha_inv_r5 + D * v * vx * x * inv_h0 / r
    result[2, 1] = _3_x_y_alpha_inv_r5 + D * v * vx * y * inv_h0 / r
    result[2, 2] = m_inv_thrust * inv_isp_g0 - D * (v + vx * vx / v)
    result[2, 3] = -D * vx * vy / v
    result[2, 4] = -m_inv_thrust * cos_sin.imag
    result[2, 6] = -m_inv_thrust * m_inv * (cos_sin.real + vx * inv_isp_g0)

    result[3, 0] = _3_x_y_alpha_inv_r5 + D * v * vy * x * inv_h0 / r
    result[3, 1] = -(r * r - 3. * y * y) * alpha_inv_r5 + D * v * vy * y * inv_h0 / r
    result[3, 2] = -D * vx * vy / v
    result[3, 3] = m_inv_thrust * inv_isp_g0 - D * (v + vy * vy / v)
    result[3, 4] = m_inv_thrust * cos_sin.real
    result[3, 6] = -m_inv_thrust * m_inv * (cos_sin.imag + vy * inv_isp_g0)

    result[4, 5] = 1.

    result[5, 6] = _3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia * (u_at_t[0] - u_at_t[1]) * m_inv * m_inv

    return result


def dgdu(u_at_t, x_at_t):
    return dgdu_tem(u_at_t, x_at_t).ravel()


def dgdx(u_at_t, x_at_t):
    return dgdx_tem(u_at_t, x_at_t).ravel()


def dhdx_zero(x_at_t):
    return dhdx_tem_zero(x_at_t).ravel()


def dhdx(x_at_t):
    return dhdx_tem(x_at_t).ravel()


def dhdx_r(x_at_t):
    return dhdx_tem_r(x_at_t).ravel()


def dhdx_v(x_at_t):
    return dhdx_tem_v(x_at_t).ravel()


def dhdx_dot(x_at_t):
    return dhdx_tem_dot(x_at_t).ravel()


def dhdx_theta1(x_at_t):
    return dhdx_tem_theta1(x_at_t).ravel()


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
#u.vector = (np.load("least_square.npy"))


u.to_vector()

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
    grad_list = [1 / 2 * grad_eval(dt * (i + 1 / (2 * np.sqrt(3)))) + 1 / 2 *
                      grad_eval(dt * (i + 1 - 1 / (2 * np.sqrt(3)))) for i in range(0, n)]
    return T / n * np.concatenate(grad_list, axis=1)


trust_fuel = optimize.LinearConstraint(np.ones(2 * n) * dt / isp / g0, 0, (16290 - 550) * kg_to_mass_unit__coeff)
trust_thrust = optimize.LinearConstraint(np.eye(2 * n), np.zeros(2 * n),
                                         16000 * 9.806 * newton_to_force_unit_coeff * np.ones(2 * n))
trust_h = optimize.NonlinearConstraint(h_constraint, np.zeros(4), np.zeros(4), jac=h_constraint_grad)
# trust_h = optimize.NonlinearConstraint(J_h.J_wrapper, 0, 0, jac=J_h.grad_wrapper)
start = chrono.time()
"""result = optimize.minimize(J.J_wrapper, u.vector, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000, "iprint": 3, "disp": True}, jac=J.grad_wrapper,
                           constraints=({"type": "ineq", "fun": thrust_constraint_min},
                                        {"type": "ineq", 'fun': thrust_constraint_max},
                                        {"type": "ineq", "fun": fuel_constraint},))
"""
"""result = optimize.minimize(J_zero.J_wrapper, u.vector, method="trust-constr", jac=J_zero.grad_wrapper,
                           hess=optimize.BFGS("skip_update"),
                           constraints=(trust_fuel, trust_thrust, trust_h),
                           options={"verbose":3})"""

"""print(result)
u1 = TimeFunction(vector=result.x, dim=2)
np.save("Results_vector_n100", u1.vector)
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
"""