import numpy as np
import matplotlib.pyplot as plt


def Euler1D(f, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros(len(t))

    x[0] = x0
    for n in range(N):
        x[n + 1] = x[n] + dt * f(x[n], t[n])

    return x, t


def Midpoint1D(f, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N
    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros(len(t))
    x[0] = x0
    for n in range(N):
        slope = f(x[n], t[n])
        tHalfstep = 0.5 * (t[n] + t[n + 1])
        xHalfstep = x[n] + dt * 0.5 * slope
        x[n + 1] = x[n] + dt * f(xHalfstep, tHalfstep)

    return x, t


def ode_test(f, x0, t0, tmax, dt, c):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N
    t = np.linspace(t0, tmax, N + 1)  # set up the times
    x = np.zeros(len(t))  # set up the x
    x[0] = x0  # initial condition
    for n in range(len(t) - 1):
        m_n = f(x[n], t[n])
        x_n_plus_half = x[n] + dt / 2 * m_n
        m_n_plus_half = f(x_n_plus_half, t[n] + dt / 2)
        x_n_plus_half_star = x[n] + dt / 2 * m_n_plus_half
        m_n_plus_half_star = f(x_n_plus_half_star, t[n] + dt / 2)
        x_n_plus_1_star = x[n] + dt * m_n_plus_half_star
        m_n_plus_1_star = f(x_n_plus_1_star, t[n] + dt)
        estimate_of_slope = (
            c[0] * m_n
            + c[1] * m_n_plus_half
            + c[2] * m_n_plus_half_star
            + c[3] * m_n_plus_1_star
        )
        x[n + 1] = x[n] + dt * estimate_of_slope
    return x, t


f = lambda x, t: -(1 / 3.0) * x + np.sin(t)
exact = lambda t: (1 / 10.0) * (19 * np.exp(-t / 3) + 3 * np.sin(t) - 9 * np.cos(t))

x0 = 1  # initial condition
t0 = 0  # initial time
tmax = 3  # max time
dt = 10.0 ** (-np.linspace(0, 1.5, 10))
C = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0.5, 0.5, 0, 0],
    [0.25, 0.5, 0, 0.25],
    [0, 0, 1, 0],
    [0, 0.5, 0.5, 0],
    [0.25, 0.25, 0.25, 0.25],
    [0.2, 0.4, 0.2, 0.2],
    [0.2, 0.2, 0.4, 0.2],
    [1 / 6, 2 / 6, 2 / 6, 1 / 6],
    [1 / 8, 3 / 8, 3 / 8, 1 / 8],
]

err_euler = np.zeros(len(dt))
err_midpoint = np.zeros(len(dt))
err_ode_test = np.zeros(len(dt))

for i in range(len(C)):
    for n in range(len(dt)):
        xeuler, t = Euler1D(f, x0, t0, tmax, dt[n])
        err_euler[n] = np.max(np.abs(xeuler - exact(t)))
        xmidpoint, t = Midpoint1D(f, x0, t0, tmax, dt[n])
        err_midpoint[n] = np.max(np.abs(xmidpoint - exact(t)))
        xtest, t = ode_test(f, x0, t0, tmax, dt[n], C[i])
        err_ode_test[n] = np.max(np.abs(xtest - exact(t)))

    plt.loglog(dt, err_euler, "r*-", dt, err_midpoint, "b*-", dt, err_ode_test, "k*-")
    plt.xlabel("Step Size")
    plt.ylabel("Error")
    plt.grid()
    plt.title(f"{i+1}-th Set of c values")
    plt.legend(["euler", "midpoint", "test method"])
    plt.show()
# Slope gives the order of the error


f = lambda x, t: -0.5*x
print(Midpoint1D(f, 6, 0, 6, 1))