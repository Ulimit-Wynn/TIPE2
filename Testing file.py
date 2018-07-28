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


def f_eval(t, u, x):
    return u.evaluate(t)


def dfdx(t, u, x):
    return 0


def dfdu(t, u, x):
    return 1


u = DifferentiableFunction(u_eval)
u.to_vector()


def g_eval(t, u, x):
    return 1 / 2 * (x(t) ** 2 + u.evaluate(t) ** 2)


def dgdx(t, u, x):
    return x(t)


def dgdu(t, u, x):
    return u.evaluate(t)


def h_eval(t, u, x):
    return 0


f = DifferentiableFunction(f=f_eval, dfdx=dfdx, dfdu=dfdu)
g = DifferentiableFunction(f=g_eval, dfdx=dgdx, dfdu=dgdu)
h = DifferentiableFunction(f=h_eval, dfdx=h_eval, dfdu=h_eval)


def J(vector):
    u = DifferentiableFunction(vector=vector, dim=1)
    u.to_func()
    def func(t, x):
        return f_eval(t, u, x)
    x_solve = sp.solve_ivp(func, (0, f.T), np.array([x0]), dense_output=True).sol

    def x(t):
        return x_solve.__call__(t)

    J = sp.quad(g.evaluate, 0, g.T, limit=50, args=(u, x))[0]
    print("J: ", J)
    return (J)


def grad_wrapper(vector):
    print("Calculating grad")
    u = DifferentiableFunction(vector=vector, dim=1)
    u.to_func()
    def func(t, x):
        return f_eval(t, u, x)
    x_solve = sp.solve_ivp(func, (0, f.T), np.array([x0]),dense_output=True).sol

    def x(t):
        return x_solve.__call__(t)

    grad = solver.gradient(u, x, f, g, h)
    print("grad calculated")
    return (grad)


result = optimize.minimize(J, u.vector, jac=grad_wrapper)

T = np.linspace(0, u.T, np.size(result.x))
print(result)
u1 = DifferentiableFunction(vector=result.x, dim=1)
u1.to_func()


def x1(t):
    def func(t, x):
        return f_eval(t, u1, x)
    x_solve = sp.solve_ivp(func, (0, f.T), np.array([x0]),dense_output=True).sol
    return x_solve.__call__(t)


def ideal_eval(t):
    return x0 / np.cosh(u.T) * np.cosh(u.T - t)


ideal = DifferentiableFunction(f=ideal_eval)
ideal.to_vector()
X = DifferentiableFunction(f=x1)
X.to_vector()
plt.plot(T, X.vector)
plt.plot(T, ideal.vector)
plt.show()
