import numpy as np

# 4.6
x0 = 1
f = lambda x: x**2 - 2
fdash = lambda x: 2 * x


def table(f, fdash, x0):
    for i in range(5):
        print(
            f"{i:<2} | x={x0:<25.10g} | f(x)={f(x0):<30.10g} | f'(x)={fdash(x0):<25.10g}"
        )
        x0 = x0 - f(x0) / fdash(x0)


# 4.7
def newton(f, fdash, x0, tol=1e-10):
    x = x0
    xnew = x - f(x) / fdash(x)
    while np.abs(x - xnew) > tol:
        if fdash(x) == 0:
            return ValueError("Derivative goes to zero")
        x = xnew
        xnew = x - f(x) / fdash(x)
    return x


print(newton(f, fdash, 1))

f1 = lambda x: x ** (1 / 3)
f1dash = lambda x: (1 / 3) * x ** (-2 / 3)
table(f1, f1dash, 7)
