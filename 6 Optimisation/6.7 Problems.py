import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as minimise

# 6.25
f1 = lambda x: x / (1 + x**4) + np.sin(x)
f2 = lambda x: (x - 1) ** 3 * (x - 2) ** 2 + np.exp(-0.5 * x)

x1 = np.linspace(-2, 2, 100)
plt.plot(x1, f1(x1))
plt.grid()
plt.show()

x2 = np.linspace(0, 2, 100)
plt.plot(x2, f2(x2))
plt.grid()
plt.show()


def GoldenSection(f, a, c, b, tol=1e-12):
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


mf1 = lambda x: -(x / (1 + x**4) + np.sin(x))
mf2 = lambda x: -((x - 1) ** 3 * (x - 2) ** 2 + np.exp(-0.5 * x))
print(GoldenSection(mf1, 0, 1, 2))
print(GoldenSection(mf2, 0, 0.6, 1))

print("- " * 40)


# 6.26
def GradientDescent(fDash, x0, alpha, tol=1e-12, MaxIterations=1000):
    x = x0
    xnew = x - alpha * fDash(x)
    count = 0
    for i in range(MaxIterations):
        x = xnew
        grad = fDash(x)
        xnew = x - alpha * grad
        count += 1
        if np.abs(xnew - x) < tol:
            return xnew, count
    raise ValueError("Does not converge")


def BruteForce(f, a, b):
    x = np.linspace(a, b, 1000)
    y = f(x)
    miny = np.min(y)
    minx = x[np.argmin(y)]
    return minx, miny


pft = lambda x: -((200 + 5 * x) * (65 - x) - 45 * x)
pftdash = lambda x: -(80 - 10 * x)
print("Brute Force")
print(
    "Min value y =", -1 * BruteForce(pft, 0, 10)[1], "at x =", BruteForce(pft, 0, 10)[0]
)
print("Golden Section Search")
print("Min at x =", GoldenSection(pft, 0, 5, 50))
print("Gradient Descent")
print("Min x at x =", GradientDescent(pftdash, 5, 0.01)[0])

print("- " * 40)

# 6.27
weight = lambda x: 800 / (1 + 3 * np.exp(-x / 30))
profit = lambda x: -(800 / (1 + 3 * np.exp(-x / 30)) * (65 - x) - 45 * x)
print("Golden Section Search")
print(
    "Min at x =",
    GoldenSection(profit, 0, 5, 50),
    "profit =",
    -1 * profit(GoldenSection(profit, 0, 5, 50)),
)

print("- " * 40)

# 6.28
# I dont have a favourite problem but its a case of making the function which i can do

# 6.29
# Cant be bothered