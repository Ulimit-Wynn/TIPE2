import scipy.integrate as sp
import scipy.optimize as optimize
from solver import DifferentiableFunction
import solver

x0 = 10


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


u = DifferentiableFunction(u_eval, dfdx=dudx)
u.to_vector()


def x(u, t):
    return sp.quad(u.evaluate, 0, t)[0] + x0


def g_eval(u):
    def func(t):
        return x(u, t) ** 2 + u.evaluate(t) ** 2

    return func


def dgdx(u):
    def func(t):
        return 2 * x(u, t)

    return func


def dgdu(u):
    def func(t):
        return 2 * u.evaluate(t)

    return func


def h_eval(t):
    return 0


def J(vector):
    u = DifferentiableFunction(vector=vector, dim=1)
    u.to_func()
    g = DifferentiableFunction(f=g_eval(u))
    return sp.quad(g.evaluate, 0, g.T)[0]


h = DifferentiableFunction(h_eval, dfdx=h_eval, dfdu=h_eval)


def grad_wrapper(vector):
    f = DifferentiableFunction(vector=vector, dim=1)
    f.to_func()
    f.dx = dudx(f)
    f.du = dudu(f)
    print(type(f.evaluate(10)))
    g = DifferentiableFunction(f=g_eval(f), dfdx=dgdx(f), dfdu=dgdu(f))
    h = DifferentiableFunction(f=h_eval, dfdx=h_eval, dfdu=h_eval)
    return solver.gradient(f, g, h)


result = optimize.minimize(J, u.vector, jac=grad_wrapper)
print(result)
