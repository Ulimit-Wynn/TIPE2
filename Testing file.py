import scipy.optimize as optimize
from Rewrite import *
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
x0 = np.array([-1, 1])
t = sympy.symbols('t')
x, y = sympy.symbols('x y')
vx, vy = sympy.symbols('vx, vy')
X = sympy.Matrix([x, y])
V = sympy.Matrix([vx, vy])
print(tuple(np.array([0, 1])))


def u_eval(t):
    return np.array([0, 0])


u = TimeFunction(u_eval)
u.to_vector()


def f_eval(time, u_at_t, x_at_t):
    return u_at_t


def g_eval(u_at_t, x_at_t):
    return 1/2 * (u_at_t.transpose() * u_at_t + x_at_t.transpose() * x_at_t)[0]


def h_eval(x_at_t):
    return 0 * x_at_t[0]


F = f_eval(t, V, X)
G = g_eval(V, X)
H = h_eval(X)
dFdX = F.jacobian(X)
dFdU = F.jacobian(V)
dGdX = G.diff(X)
dGdU = G.diff(V)
dHdX = H.diff(X)


dfdx = sympy.lambdify((t, V, X), dFdX)
dfdu = sympy.lambdify((t, V, X), dFdU)
dgdx_lam = sympy.lambdify((V, X), dGdX)
dgdu_lam = sympy.lambdify((V, X), dGdU)
dhdx_lam = sympy.lambdify((X,), dHdX)


def dhdx(x_at_t):
    return dhdx_lam(x_at_t).ravel()


def dgdx(u_at_t, x_at_t):
    return dgdx_lam(u_at_t, x_at_t).ravel()


def dgdu(u_at_t, x_at_t):
    return dgdu_lam(u_at_t, x_at_t).ravel()


f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=dhdx)
system = DynamicalSystem(f, x0)
J = Functional(system, g, h)
print(h.dx(np.array([1, 2])))

result = optimize.minimize(J.J_wrapper, u.vector, jac=J.grad_wrapper, tol=10 ** (-5))
time = np.linspace(0, T, n)
print(result)
u1 = TimeFunction(vector=result.x, dim=2)
u1.to_func()


P = TimeFunction(f=system.solve(u1))
P.to_vector()
X1 = P.vector[::2]
X2 = P.vector[1::2]
plt.plot(time, X1)
plt.plot(time, X2)
plt.show()
end = time.time()
print((end - start)/60)
