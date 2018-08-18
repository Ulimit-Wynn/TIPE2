import scipy.optimize as optimize
from Rewrite import *
import numpy as np
import matplotlib.pyplot as plt
import sympy
import time as chrono


x0 = np.array([1])
t = sympy.symbols('t')
x = sympy.symbols('x')
Vx = sympy.symbols('Vx')
X = sympy.Matrix([x])
U = sympy.Matrix([Vx])


F = U
H = 0 * x
G = 1 / 2 *(U.transpose() * U)[0] + (X.transpose() * X)[0]
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
print(f_eval(0.2, np.array([0.1]), np.array([1])))


def dfdu(t, u_at_t, x_at_t):
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
print(f.evaluate(0, np.array([1]), np.array([1])))

def u_eval(t):
    return np.array([0])


time = np.linspace(0, T, n)
u = TimeFunction(u_eval)
u.to_vector()
start = chrono.time()
result = optimize.minimize(J.J_wrapper, u.vector, jac=J.grad_wrapper, tol=10 ** (-13))
time = np.linspace(0, T, n)
print(result)
u1 = TimeFunction(vector=result.x, dim=1)
u1.to_func()

P = TimeFunction(f=system.solve(u1))
P.to_vector()
X1 = P.vector
print(P(T))
plt.figure(1)
plt.plot(time, X1)
plt.figure(4)
plt.plot(time, u1.vector)
end = chrono.time()
print("time: ", end - start)
plt.show()