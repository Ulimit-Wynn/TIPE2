import scipy.optimize as optimize
from Rewrite import *
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([1, 1])


def u_eval(t):
    return np.array([0, 1])


u = TimeFunction(u_eval)
u.to_vector()


def f_eval(t, u_at_t, x_at_t):
    return u_at_t


def dfdx(t, u_at_t, x_at_t):
    return np.array([[0, 0], [0, 0]])


def dfdu(t, u_at_t, x_at_t):
    return np.array([[1, 0], [0, 1]])


def g_eval(u_at_t, x_at_t):
    return 1 / 2 * (x_at_t @ x_at_t + u_at_t @ u_at_t)


def dgdx(u_at_t, x_at_t):
    return x_at_t


def dgdu(u_at_t, x_at_t):
    return u_at_t


def h_eval(x_at_t):
    return 0

def dhdx(x_at_t):
    return np.array([0, 0])

def dhdu(x_at_t):
    return np.array([0, 0])


f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=dhdx, dfdu=dhdu)
system = DynamicalSystem(f, x0)
J = Functional(system, g, h)
result = optimize.minimize(J.J_wrapper, u.vector, jac=J.grad_wrapper, tol=10 ** (-5))
time = np.linspace(0, T, n)
print(result)
u1 = TimeFunction(vector=result.x, dim=1)
u1.to_func()


X = TimeFunction(f=system.solve(u1))
X.to_vector()
X1 = X.vector[::2]
X2 = X.vector[1::2]
plt.plot(time, X1)
plt.plot(time, X2)
plt.show()
