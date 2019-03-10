from Solver import *
import math, cmath
import sympy


_3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia = .75 * meter_to_distance_unit_coeff / rocket_inertia
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
G = (thrust / (isp * g0))
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


