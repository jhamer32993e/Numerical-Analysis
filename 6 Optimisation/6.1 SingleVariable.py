import matplotlib.pyplot as plt
import numpy as np

# 6.2
x = np.linspace(0, 1.5, 1000)
y = lambda x: -np.exp(-(x**2)) - np.sin(x**2)
plt.plot(x, y(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, which="both", ls="--")
plt.show()

y = y(x)
print(x[np.argmin(y)])


# 6.5
def GoldenSection(f, a, b, c, tol=1e-12):
    if a >= c or c >= b:
        raise ValueError("points not in order a < c < b")

    if f(a) <= f(c) or f(b) <= f(c):
        raise ValueError("f(c) not less than both f(a) and f(b)")
    rho = (3 - 5**0.5) / 2
    fc = f(c)
    while b - a >= tol:
        if b - c > c - a:
            x = c + rho * (b - c)
            fx = f(x)

            if fx < fc:
                a = c
                c = x
                fc = fx
            else:
                b = x

        else:
            x = c - rho * (c - a)
            fx = f(x)
            if fx < fc:
                b = c
                c = x
                fc = fx
            else:
                a = x

    return (a + b) / 2


f1 = lambda x: -np.exp(-(x**2)) - np.sin(x**2)
print(GoldenSection(f1, 0, 2, 1))
