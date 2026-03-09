import numpy as np


def SecondOrderTaylorSolve(f, fdash, fddash, x0, tol=1e-10):
    xp = x0
    # Positive root
    while abs(f(xp)) > tol:
        a = 0.5 * fddash(xp)
        b = fdash(xp) - xp * fddash(xp)
        c = f(xp) - fdash(xp) * xp + 0.5 * fddash(xp) * (xp**2)
        xp = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return xp


f = lambda x: x**2 - 2
fdash = lambda x: 2 * x
fddash = lambda x: 2 * x**0
print(SecondOrderTaylorSolve(f, fdash, fddash, 1))
