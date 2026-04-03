import numpy as np

# 3.22
f1 = lambda x: np.cos(x)
x1 = 0.2
for i in range(10):
    print(x1)
    x1 = np.cos(x1)
print(x1)
print("Estimated solution of cos(x)-x=0 is:", x1)
print("ie. Fixed point of cos(x)")
print("- " * 40)


def fixed_point(g, x0, tol, n):
    y = 0
    for i in range(n):
        y = x0
        x0 = g(x0)
        if np.abs(y - x0) < tol:
            return x0
            break
    return ValueError("Iteration has not converged")


g1 = lambda x: np.cos(x)
print(fixed_point(g1, 2, 1e-8, 100))
