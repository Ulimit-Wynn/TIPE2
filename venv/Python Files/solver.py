import scipy.integrate as integrate
import numpy as np


class DifferentiableFunction:
    T = 360
    dt = 10
    def __init__(f, dfdx, dfdu)
        self.evaluate = f
        self.dx = dfdx
        self.du = dfdu
    def to_vector(self):




def gradient(f, g, h):
    # solve ode p'(t) = -f.dx(t) p(t) - g.dx
    def func(t, y):
        return f.dx(t) @ y + g.dx(T)

    p = integrate.solve_ivp(func, (0, T), h.dx(T)).y
    temp = p
    for i in range(0, np.size(p, 0)):
        temp[np.size(p) - i] = p[i]
    p = temp
    def grad_eval(t):
        return f.dx(t) @ p(t)

    grad = DifferentiableFunction(grad_eval, dfdx=None, dfdu=None)