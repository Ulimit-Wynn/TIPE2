import scipy.optimize as optimize
from Rewrite import *
import numpy as np
import matplotlib.pyplot as plt
import sympy
import time as chrono

start = chrono.time()
x0 = np.array([6371000, 0, 0, 0, 549000])
t = sympy.symbols('t')
x, y, vx, vy, m = sympy.symbols('x y vx vy m')
Fx, Fy = sympy.symbols('Fx Fy')
X = sympy.Matrix([x, y, vx, vy, m])
U = sympy.Matrix([Fx, Fy])
r = sympy.sqrt(x ** 2 + y ** 2)
r1 = (x * vx + y * vy) / r
theta = sympy.atan2(y, x)
theta1 = (vy * x - vx * y) / (x ** 2 + y ** 2)

Force = sympy.sqrt(Fx ** 2 + Fy ** 2)
ax = Fx/m - M * G * x / (r ** 3) - vx * Force / (g * isp * m)
ay = Fy/m - M * G * y / (r ** 3) - vy * Force / (g * isp * m)
mdot = -Force / (g * isp)


e = (r ** 2 * theta1) / (G * M) * sympy.sqrt((r ** 2 * theta1) ** 2 / (r ** 2) + r1 ** 2)
a = ((r ** 2 * theta1) ** 2 / (G * M)) / (1 - e ** 2)


F = sympy.Matrix([vx, vy, ax, ay, mdot])
#H = (((e - e0) ** 2) / (e0 ** 2) + ((a - a0) ** 2) / (a0 ** 2))
H = (((x - 700000) ** 2)/(7000000 ** 2) + ((y - 250000) ** 2) / (250000 ** 2))
G = 0 * (((U.transpose() * U)[0]) / ((549000 * T) ** 2))
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


def f_eval(t, u_at_t, x_at_t):
    return f_tem(t, u_at_t, x_at_t).ravel()


def dfdu(t, u_at_t, x_at_t):
    if u_at_t[0] == 0 and u_at_t[1] == 0:
        return np.array([[0, 0], [0, 0], [1/x_at_t[4], 0], [0, 1/x_at_t[4]], [0, 0]])
    return dfdu_tem(t, u_at_t, x_at_t)


def dgdu(u_at_t, x_at_t):
    return dgdu_tem(u_at_t, x_at_t).ravel()


def dgdx(u_at_t, x_at_t):
    return dgdx_tem(u_at_t, x_at_t).ravel()


def dhdx(x_at_t):
    return dhdx_tem(x_at_t).ravel()

f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=dhdx)
system = DynamicalSystem(f, x0)
J = Functional(system, g, h)


def u_eval(t):
    if t<T/2.72:
        return np.array([6804000, 0])
    return np.array([0, 617000])

u = TimeFunction(u_eval)
u.to_vector()
result = optimize.minimize(J.J_wrapper, u.vector, jac=J.grad_wrapper, tol=10 ** (-20))
time = np.linspace(0, T, n)
print(result)
u1 = TimeFunction(vector=result.x, dim=2)
u1.to_func()


P = TimeFunction(f=system.solve(u1))
P.to_vector()
print(dfdu(T, u1(T), P(T)))
X1 = P.vector[::5]
X2 = P.vector[1::5]
print(P(T))
plt.figure(1)
plt.plot(time, X1)
plt.figure(2)
plt.plot(time, X2)
plt.figure(3)
plt.plot(X1, X2)
plt.figure(4)
plt.plot(time, u1.vector[::2])
plt.figure(5)
plt.plot(time, u1.vector[1::2])
#end = chrono.time()
plt.show()
