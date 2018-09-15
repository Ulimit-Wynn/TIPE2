import scipy.optimize as optimize
from Solver import *
import numpy as np
import matplotlib.pyplot as plt
import sympy
import time as chrono

alpha = 1
beta = 1
x0 = np.array([1, 0, 0, 1, 1])
t = sympy.symbols('t')
x, y, vx, vy, m = sympy.symbols('x y vx vy m')
thrust, theta = sympy.symbols('thrust theta')
X = sympy.Matrix([x, y, vx, vy, m])
U = sympy.Matrix([thrust, theta])
r = sympy.sqrt(x ** 2 + y ** 2)
r1 = (x * vx + y * vy) / r
phi = sympy.atan2(y, x)
phi1 = (vx * y - x * vy) / (x ** 2 + y ** 2)
thrust_matrix = np.zeros((n, 2 * n))
fuel_matrix = np.zeros(2 * n)
for i in range(0, n):
    thrust_matrix[i][2 * i] = 1
for i in range(0, n):
    fuel_matrix[2 * i] = dt / beta
print(fuel_matrix)

ax = thrust * sympy.cos(theta) / m - alpha * x / (r ** 3) + vx * thrust / (beta * m)
ay = thrust * sympy.sin(theta) / m - alpha * y / (r ** 3) + vy * thrust / (beta * m)
mdot = -thrust / beta

energy = m * (1 / 2 * (vx ** 2 + vy ** 2) - alpha / r)
moment = m * (vx * y - x * vy)

F = sympy.Matrix([vx, vy, ax, ay, mdot])

H = x
G = thrust
print(F)
print(G)
print(H)
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


def f_eval(time, u_at_t, x_at_t):
    return f_tem(time, u_at_t, x_at_t).ravel()


def dfdu(time, u_at_t, x_at_t):
    if u_at_t[0] == 0:
        return np.array([[0, 0], [0, 0], [1 / x_at_t[4], 0], [0, 1 / x_at_t[4]], [0, 0]])
    return dfdu_tem(time, u_at_t, x_at_t)


def dgdu(u_at_t, x_at_t):
    return dgdu_tem(u_at_t, x_at_t).ravel()


def dgdx(u_at_t, x_at_t):
    return dgdx_tem(u_at_t, x_at_t).ravel()


def dhdx(x_at_t):
    return dhdx_tem(x_at_t).ravel()


def u_eval(time):
    return np.array([0, 0])


def thrust_constraint(vector):
    res = thrust_matrix @ vector
    return res


def fuel_constraint(vector):
    return 0.5 - np.sum(vector[::2]) * dt / beta

f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=dhdx)
system = DynamicalSystem(f, x0)
J = Functional(system, g, h)
u = TimeFunction(u_eval)
u.to_vector()

start = chrono.time()
result = optimize.minimize(J.J_wrapper, u.vector, jac=J.grad_wrapper, tol=1e-10, constraints=[{"type": "ineq", "fun": thrust_constraint}, {"type": "ineq", "fun": fuel_constraint}])
time_array = np.linspace(0, T, n)
print(result)
u1 = TimeFunction(vector=result.x, dim=2)
u1.to_func()
grad = result.jac
print("final grad: ", grad)
d = np.ones(np.size(u.vector))
for i in range(2, 10):
    print(" ")
    print("eps=", "10^-", str(i))
    eps = 10 ** (-1 * i)
    u_test1 = TimeFunction(vector=u1.vector + eps * d, dim=u.dim)
    u_test2 = TimeFunction(vector=u1.vector - eps * d, dim=u.dim)
    print("d @ grad: ", d @ grad)
    print("Finite difference: ", (J(u_test1) - J(u_test2)) / 2 / eps)

P = TimeFunction(f=system.solve(u1))
P.to_vector()
print(u1(2.01))
print(P(2.01))
print(f.evaluate(2.01, u1(2.01), P(2.01)))
X1 = P.vector[::5]
X2 = P.vector[1::5]
gr1 = grad[::2]
gr2 = grad[1::2]
print(P(T))
plt.figure(1)
plt.plot(time_array, gr1)
plt.figure(2)
plt.plot(time_array, gr2)
plt.figure(3)
plt.plot(X1, X2)
plt.figure(4)
plt.plot(time_array, u1.vector[::2])
plt.figure(5)
plt.plot(time_array, u1.vector[1::2])
end = chrono.time()
print("time: ", end - start)
plt.show()
