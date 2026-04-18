import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import optimize as optimise
from IPython.display import HTML


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


# 8.22
A = -(10 ** np.arange(0, 7, dtype=float))
x0 = 1.5  # initial condition
t0 = 0  # initial time
tmax = 1  # max time
DT = 10.0 ** (-np.linspace(0, 1, 10))

for i, a in enumerate(A):
    f = lambda x, t: a * (x - np.cos(t)) - np.sin(t)
    exact = lambda t: np.cos(t) + 0.5 * np.exp(a * t)

    err_euler = np.zeros(len(DT))
    err_midpoint = np.zeros(len(DT))
    err_rk4 = np.zeros(len(DT))

    for j, dt in enumerate(DT):
        xEuler, t = Euler1D(f, x0, t0, tmax, dt)
        err_euler[j] = np.max(abs(xEuler - exact(t)))
        xMidpoint, t = Midpoint1D(f, x0, t0, tmax, dt)
        err_midpoint[j] = np.max(abs(xMidpoint - exact(t)))
        xRK4, t = RK4_1D(f, x0, t0, tmax, dt)
        err_rk4[j] = np.max(abs(xRK4 - exact(t)))

    plt.loglog(DT, err_euler, "g-", label="Euler")
    plt.loglog(DT, err_midpoint, "b-", label="Midpoint")
    plt.loglog(DT, err_rk4, "r-", label="RK4")
    plt.legend()
    plt.xlabel("Step size")
    plt.ylabel("Maximum global error")
    plt.title(f"Lambda = {a}")
    plt.grid()
    plt.show()

print("- " * 40)


# 8.23
def BackwardEuler(F, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0

    for n in range(N):
        G = lambda y: y - dt * F(y, t[n + 1]) - x[n, :]
        x[n + 1, :] = optimise.fsolve(G, x[n, :])

    return x, t


F = lambda x, t: np.array(
    [
        x[1],
        -x[0] / (x[0] ** 2 + x[2] ** 2) ** 1.5,
        x[3],
        -x[2] / (x[0] ** 2 + x[2] ** 2) ** 1.5,
    ]
)
t0 = 0
tmax = 100
dt = 0.1

for i in range(1, 6):
    x0 = [4, 0, 0, i / 2]
    x, t = BackwardEuler(F, x0, t0, tmax, dt)

    plt.plot(x[:, 0], x[:, 2], label=f"v_y(0) = {i/2}")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Orbital Dynamics via Implicit Backward Euler")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()

# Animation by Gemini
# Uncomment to show animation, highlight then "CTRL + /"
# Takes a while to run
# x0 = [4.0, 0.0, 0.0, 0.5]
# x_sol, t_sol = BackwardEuler(F, x0, t0, tmax, dt)
# x_data = x_sol[:, 0]
# y_data = x_sol[:, 2]

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)
# ax.plot(0, 0, "yo", markersize=15, label="Sun")
# ax.set_title("Backward Euler Orbital Animation")
# ax.grid()
# ax.legend()

# (trail,) = ax.plot([], [], "b-", alpha=0.5)
# (planet,) = ax.plot([], [], "ro", markersize=8)


# def init():
#     trail.set_data([], [])
#     planet.set_data([], [])
#     return trail, planet


# def animate(i):
#     trail.set_data(x_data[:i], y_data[:i])
#     planet.set_data([x_data[i]], [y_data[i]])
#     return trail, planet


# ani = animation.FuncAnimation(
#     fig, animate, init_func=init, frames=len(x_data), interval=20, blit=True
# )

# plt.close(fig)
# HTML(ani.to_jshtml())

print("- " * 40)

# 8.24
