import numpy as np
import matplotlib.pyplot as plt


# 8.5
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


# 8.6
def RK4Error(f, x0, t0, tmax, hFactor, xExact):
    dt = hFactor ** (-np.linspace(0, 4, 50))
    errors = np.zeros_like(dt)
    for i in range(len(dt)):
        xApprox, t = RK4_1D(f, x0, t0, tmax, dt[i])
        errors[i] = np.max(np.abs(xExact(t) - xApprox))

    plt.loglog(dt, errors)
    plt.xlabel("Step size")
    plt.ylabel("Maximum error")
    plt.grid()
    plt.show()


# 8.7
def RK4(F, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)

    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
    for n in range(N):
        m_n = F(x[n, :], t[n])
        x_n_plus_half = x[n, :] + dt / 2 * m_n
        m_n_plus_half = F(x_n_plus_half, t[n] + dt / 2)
        x_n_plus_half_star = x[n, :] + dt / 2 * m_n_plus_half
        m_n_plus_half_star = F(x_n_plus_half_star, t[n] + dt / 2)
        x_n_plus_1_star = x[n, :] + dt * m_n_plus_half_star
        m_n_plus_1_star = F(x_n_plus_1_star, t[n] + dt)
        estimate_of_slope = (
            1 / 6 * m_n
            + 2 / 6 * m_n_plus_half
            + 2 / 6 * m_n_plus_half_star
            + 1 / 6 * m_n_plus_1_star
        )
        x[n + 1, :] = x[n, :] + dt * estimate_of_slope
    return x, t
