import scipy.integrate as sp
import scipy.optimize as optimize
from solver import DifferentiableFunction
import numpy as np
import matplotlib.pyplot as plt
import solver

x0 = 1


def u_eval(t):
    return 0


def dudx(u):
    def func(t):
        return (u.evaluate(t) - u.evaluate(t + 0.00001)) / 0.00001

    return func


def dudu(u):
    def func(t):
        return 1

    return func


u = DifferentiableFunction(u_eval)
u.to_vector()


def x(u, t):
    return sp.quad(u.evaluate, 0, t, limit=100)[0] + x0


def g_eval(u):
    def func(t):
        return 1 / 2 * (x(u, t) ** 2 + u.evaluate(t) ** 2)

    return func


def dgdx(u):
    def func(t):
        return x(u, t)

    return func


def dgdu(u):
    def func(t):
        return u.evaluate(t)

    return func


def h_eval(t):
    return 0


def J(vector):
    u = DifferentiableFunction(vector=vector, dim=1)
    u.to_func()
    g = DifferentiableFunction(f=g_eval(u))
    J = sp.quad(g.evaluate, 0, g.T, limit=50)[0]
    print("J: ", J)
    return (J)


h = DifferentiableFunction(h_eval, dfdx=h_eval, dfdu=h_eval)


def grad_wrapper(vector):
    print("Calculating grad")
    f = DifferentiableFunction(vector=vector, dim=1)
    f.to_func()
    f.dx = dudx(f)
    f.du = dudu(f)
    g = DifferentiableFunction(f=g_eval(f), dfdx=dgdx(f), dfdu=dgdu(f))
    h = DifferentiableFunction(f=h_eval, dfdx=h_eval, dfdu=h_eval)
    grad = solver.gradient(f, g, h)
    return (grad)


result = optimize.minimize(J, u.vector, jac=grad_wrapper)

T = np.linspace(0, u.T, np.size(result.x))
print(result)
u1 = DifferentiableFunction(vector=result.x, dim=1)
u1.to_func()


def x1(t):
    return x(u1, t)

def ideal_eval(t):
    return x0/np.cosh(u.T) * np.cosh(u.T - t)
ideal = DifferentiableFunction(f=ideal_eval)
ideal.to_vector()
X = DifferentiableFunction(f=x1)
X.to_vector()
plt.plot(T, X.vector)
plt.plot(T, ideal.vector)
plt.show()
