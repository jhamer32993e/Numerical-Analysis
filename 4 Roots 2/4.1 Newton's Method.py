import numpy as np

# 4.6
x0 = 1
f = lambda x: x**2 - 2
fdash = lambda x: 2 * x
for i in range(5):
    print(f"{i:<2} | x={x0:<15.10g} | f(x)={f(x0):<15.10g} | f'(x)={fdash(x0):<15.10g}")
    x0 = x0 - f(x0) / fdash(x0)

#4.7
def newton(f, fdash, x0, tol=1e-10):
    x = x0
    xnew = x - f(x) / fdash(x)
    while(np.abs(x-xnew) > tol):
        x = xnew
        xnew = x - f(x) / fdash(x)
    return x

print(newton(f, fdash, 1))