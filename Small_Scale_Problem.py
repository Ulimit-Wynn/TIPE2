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
    fuel_matrix[2 * i] = dt/beta

ax = thrust * sympy.cos(theta) /m - alpha * x / (r ** 3) - vx * thrust / (beta * m)
ay = thrust * sympy.sin(theta) /m - alpha * y / (r ** 3) - vy * thrust / (beta * m)
mdot = -thrust / (beta)


#e = (r ** 2 * phi1) / (alpha) * sympy.sqrt((r ** 2 * phi1) ** 2 / (r ** 2) + + r1 ** 2)
#a = ((r ** 2 * phi1) ** 2 / (alpha)) / (1 - e ** 2)


F = sympy.Matrix([vx, vy, ax, ay, mdot])
H = ((x - 2) ** 2 + (y - 1) ** 2)
G = 0 * (U.transpose() * U)[0]
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

print(dFdU)
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


def u_eval(t):
    return np.array([1, 1])

def thrust_constraint(x):
    return multiply(thrust_matrix, x)


def fuel_constraint(x):
    return 1 - fuel_matrix @ x

print('fuel: ', fuel_constraint(np.ones(2 * n)))



f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=dhdx)
system = DynamicalSystem(f, x0)
J = Functional(system, g, h)
time = np.linspace(0, T, n)
u = TimeFunction(u_eval)
u.to_vector()


start = chrono.time()
result = optimize.minimize(J.J_wrapper, u.vector, jac=J.grad_wrapper, constraints=[{"type":"ineq", "fun":thrust_constraint}, {"type":"ineq", "fun":fuel_constraint}])
time = np.linspace(0, T, n)
print(result)
u1 = TimeFunction(vector=result.x, dim=2)
u1.to_func()

P = TimeFunction(f=system.solve(u1))
P.to_vector()
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
end = chrono.time()
print("time: ", end - start)
plt.show()
