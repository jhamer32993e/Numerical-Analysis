import numpy as np


def bisection(f, a, b, tol=1e-10):
    if f(a) * f(b) >= 0:
        raise ValueError(
            "f(a), f(b) do not have opposite signs, no guarantee of a root"
        )
    midpoint = (a + b) / 2
    while abs(a - b) > 2 * tol:
        if f(a) * f(midpoint) < 0:
            b = midpoint
        elif f(b) * f(midpoint) < 0:
            a = midpoint
        elif f(midpoint) == 0:
            return midpoint
        midpoint = (a + b) / 2
    return midpoint


def newton(f, fdash, x0, tol=1e-10):
    x = x0
    xnew = x - f(x) / fdash(x)
    for i in range(30):
        if fdash(x) == 0:
            return ValueError("Derivative goes to zero")
        x = xnew
        xnew = x - (f(x) / fdash(x))
        if np.abs(x - xnew) > tol:
            return x
    return "Does not converge"


def secant(f, x0, x1, tol=1e-10):
    xnm1 = x0
    xn = x1
    while np.abs(xn - xnm1) > tol:
        xnp1 = xn - f(xn) * (xn - xnm1) / (f(xn) - f(xnm1))
        xnm1 = xn
        xn = xnp1
    return xn


f = lambda x: 3 * np.sin(x) + 9 - x**2 + np.cos(x)
fdash = lambda x: 3 * np.cos(x) - 2 * x - np.sin(x)


def comparison(f, fdash, a, b, x0, x1):
    print("Bisection: ", bisection(f, a, b))
    print("Newton: ", newton(f, fdash, x0))
    print("Secant: ", secant(f, x0, x1))
    return


comparison(f, fdash, 2, 4, 3, 4)
