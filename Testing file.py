import scipy.optimize as optimize
from Rewrite import *
import numpy as np
import matplotlib.pyplot as plt


x0 = np.array([1])

def u_eval(t):
    return 0

u = TimeFunction(u_eval)
u.to_vector()


def f_eval(t, u_at_t, x_at_t):
    return u_at_t


def dfdx(t, u_at_t, x_at_t):
    return 0


def dfdu(t, u_at_t, x_at_t):
    return 1


def g_eval(u_at_t, x_at_t):
    return 1 / 2 * (x_at_t ** 2 + u_at_t ** 2)


def dgdx(u_at_t, x_at_t):
    return x_at_t


def dgdu(u_at_t, x_at_t):
    return u_at_t


def h_eval(x_at_t):
    return 0



f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=h_eval, dfdu=h_eval)

system = DynamicalSystem(f, x0)
J = Functional(system, g, h)
result = optimize.minimize(J.J_wrapper, u.vector, jac=J.grad_wrapper)
time = np.linspace(0, T, np.size(result.x))
print(result)
u1 = TimeFunction(vector=result.x, dim=1)
u1.to_func()


def ideal_eval(t):
    return x0 / np.cosh(T) * np.cosh(T - t)

def ideal_u(t):
    return -x0 / np.cosh(T) * np.sinh(T - t)


ideal = TimeFunction(f=ideal_eval)
ideal.to_vector()
u_ideal = TimeFunction(f=ideal_u)
u_ideal.to_vector()
X = TimeFunction(f=system.solve(u1))
X.to_vector()
plt.plot(time, u1.vector)
plt.plot(time, u_ideal.vector)
plt.show()
