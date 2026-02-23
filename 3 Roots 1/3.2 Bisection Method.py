import numpy as np


# 3.10
def bisection(f, a, b, tol=1e-5):
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


# 3.11
f1 = lambda x: x**2 - 2
print(bisection(f1, 0, 2))

f2 = lambda x: np.sin(x) + x**2 - 2 * np.log(x) - 5
print(bisection(f2, 1, 5))

f3 = lambda x: 3 * np.sin(x) + 9 - x**2 - np.cos(x)
print(bisection(f3, 1, 5))
