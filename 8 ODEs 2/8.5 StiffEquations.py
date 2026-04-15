import numpy as np
import matplotlib.pyplot as plt


# 8.15
def Euler1D(f, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros(len(t))

    x[0] = x0
    for n in range(N):
        x[n + 1] = x[n] + dt * f(x[n], t[n])

    return x, t


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


def RK4_1D(f, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N
    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros(len(t))
    x[0] = x0
    for n in range(N):
        m_n = f(x[n], t[n])
        x_n_plus_half = x[n] + dt / 2 * m_n
        m_n_plus_half = f(x_n_plus_half, t[n] + dt / 2)
        x_n_plus_half_star = x[n] + dt / 2 * m_n_plus_half
        m_n_plus_half_star = f(x_n_plus_half_star, t[n] + dt / 2)
        x_n_plus_1_star = x[n] + dt * m_n_plus_half_star
        m_n_plus_1_star = f(x_n_plus_1_star, t[n] + dt)
        estimate_of_slope = (
            1 / 6 * m_n
            + 2 / 6 * m_n_plus_half
            + 2 / 6 * m_n_plus_half_star
            + 1 / 6 * m_n_plus_1_star
        )
        x[n + 1] = x[n] + dt * estimate_of_slope
    return x, t


f = lambda x, t: -100 * x + np.sin(t)
exact = (
    lambda t: 1 / 10001 * np.exp(-100 * t)
    + 100 / 10001 * np.sin(t)
    - 1 / 10001 * np.cos(t)
)
H = [0.1, 0.01, 0.001]
x0 = 0
t0 = 0
tmax = 1

for i in H:
    EulerX, t = Euler1D(f, x0, t0, tmax, i)
    plt.plot(t, EulerX, "g-", label="Forward Euler")
    BackwardEulerX, t = BackwardEuler1D(f, x0, t0, tmax, i)
    plt.plot(t, BackwardEulerX, "b-", label="Backwards Euler")
    RK4X, t = RK4_1D(f, x0, t0, tmax, i)
    plt.plot(t, RK4X, "r-", label="RK4")
    plt.plot(t, exact(t), label="Exact")
    plt.legend()
    plt.grid()
    plt.title(f"dt = {i}")
    plt.ylim(-1, 1)
    plt.show()
