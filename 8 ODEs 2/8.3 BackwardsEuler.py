import numpy as np
import matplotlib.pyplot as plt


# 8.10
def secant(f, x0, x1, tol=1e-10):
    xnm1 = x0
    xn = x1
    while np.abs(xn - xnm1) > tol:
        xnp1 = xn - f(xn) * (xn - xnm1) / (f(xn) - f(xnm1))
        xnm1 = xn
        xn = xnp1
    return xn


def BackwardEuler1D(f, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N
    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros(len(t))
    x[0] = x0

    for i in range(N):
        G = lambda y: y - x[i] - dt * f(y, t[i + 1])
        x[i + 1] = secant(G, x[i], x[i] + dt * f(x[i], t[i]))

    return x, t


# 8.11
def Euler1D(f, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros(len(t))

    x[0] = x0
    for n in range(N):
        x[n + 1] = x[n] + dt * f(x[n], t[n])

    return x, t


f = lambda x, t: -x / 3 + np.sin(t)
xExact = lambda t: (19 * np.exp(-t / 3) + 3 * np.sin(t) - 9 * np.cos(t)) / 10
x0 = 1
t0 = 0
tmax = 4
dt = 0.1
xEuler, t = Euler1D(f, x0, t0, tmax, dt)
xBackward, t = BackwardEuler1D(f, x0, t0, tmax, dt)

plt.plot(t, xEuler, "g-.", label="Euler")
plt.plot(t, xBackward, "b--", label="Backwards")
plt.plot(t, xExact(t), "r-", label="Exact")
plt.legend()
plt.show()
