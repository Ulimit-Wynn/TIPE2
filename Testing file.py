import scipy.integrate as sp
import scipy.optimize as optimize
from solver import DifferentiableFunction
import numpy as np
import matplotlib.pyplot as plt
import solver

x0 = 1
T = DifferentiableFunction.T

def u_eval(t):
    return 0


def f_eval(u_at_t, x_at_t):
    return u_at_t


def dfdx(u_at_t, x_at_t):
    return 0


def dfdu(u_at_t, x_at_t):
    return 1


u = DifferentiableFunction(u_eval)
u.to_vector()


def g_eval(u_at_t, x_at_t):
    return 1 / 2 * (x_at_t ** 2 + u_at_t ** 2)


def dgdx(u_at_t, x_at_t):
    return x_at_t


def dgdu(u_at_t, x_at_t):
    return u_at_t


def h_eval(u_at_t, x_at_t):
    return 0


f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=h_eval, dfdu=h_eval)


def J_wrapper(vector):
    u = DifferentiableFunction(vector=vector, dim=1)
    u.to_func()
    j = solver.J(u, f, g, h)
    return j


def grad_wrapper(vector):
    print("Calculating grad")
    u = DifferentiableFunction(vector=vector, dim=1)
    u.to_func()
    grad = solver.gradient(u, f, g, h)
    def grad2(t):
        return grad.evaluate(t) ** 2
    print(np.sqrt(sp.quad(grad2, 0, T)[0]))
    print("grad calculated")
    grad.to_vector()
    return (grad.vector)


result = optimize.minimize(J_wrapper, u.vector, jac=grad_wrapper)

time = np.linspace(0, u.T, np.size(result.x))
print(result)
u1 = DifferentiableFunction(vector=result.x, dim=1)
u1.to_func()


def x1(t):
    def func(t, x):
        return f_eval(u1.evaluate(t), x)
    x_solve = sp.solve_ivp(func, (0, f.T), np.array([x0]), dense_output=True).sol
    return x_solve.__call__(t)


def ideal_eval(t):
    return x0 / np.cosh(u.T) * np.cosh(u.T - t)


ideal = DifferentiableFunction(f=ideal_eval)
ideal.to_vector()
X = DifferentiableFunction(f=x1)
X.to_vector()
plt.plot(time, X.vector)
plt.plot(time, result.x)
plt.show()
