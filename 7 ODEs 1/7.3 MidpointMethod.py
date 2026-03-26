import numpy as np
import matplotlib.pyplot as plt


# 7.14
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


# 7.15
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
x0 = 1
t0 = 0
tmax = 10
dt = 1
xExact = lambda t: 0.1 * (19 * np.exp(-t / 3) + 3 * np.sin(t) - 9 * np.cos(t))
x, t = Euler1D(f, x0, t0, tmax, dt)
plt.plot(t, x, "b-", label="Euler")

x, t = Midpoint1D(f, x0, t0, tmax, dt)
plt.plot(t, x, "m-", label="Midpoint")

t_highres = np.linspace(t0, tmax, 100)
plt.plot(t_highres, xExact(t_highres), "r--", label="Exact")
plt.legend()
plt.grid()
plt.show()


print("- " * 40)


# 7.16
def MidpointError(f, x0, t0, tmax, hFactor, xExact):
    dt = hFactor ** (-np.linspace(0, 4, 50))
    errors = np.zeros_like(dt)
    for i in range(len(dt)):
        xApprox, t = Midpoint1D(f, x0, t0, tmax, dt[i])
        errors[i] = np.max(np.abs(xExact(t) - xApprox))

    plt.loglog(dt, errors)
    plt.xlabel("Step size")
    plt.ylabel("Maximum error")
    plt.grid()
    plt.show()


MidpointError(f, x0, t0, tmax, 10, xExact)

print("- " * 40)


# 7.18
def Midpoint(F, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)

    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0

    for n in range(N):
        slope = F(x[n, :], t[n])

        tHalfstep = 0.5 * (t[n] + t[n + 1])
        xHalfstep = x[n, :] + 0.5 * dt * slope

        x[n + 1, :] = x[n, :] + dt * F(xHalfstep, tHalfstep)

    return x, t


a, b, c, d = 1, 0.1, 1, 0.1
F = lambda x, t: np.array([a * x[0] - b * x[0] * x[1], d * x[0] * x[1] - c * x[1]])
x0 = [10, 5]
t0 = 0
tmax = 20
dt = 0.001
x, t = Midpoint(F, x0, t0, tmax, dt)

plt.plot(t, x[:, 0], "b-", t, x[:, 1], "r--")
plt.grid()
plt.title("Time Evolution of Prey and Predators")
plt.legend(["Prey", "Predators"])
plt.xlabel("Time")
plt.ylabel("Prey and Predators")
plt.show()

plt.plot(x[:, 0], x[:, 1])
plt.grid()
plt.title("Phase Plot")
plt.ylabel("Predators")
plt.xlabel("Prey")
plt.show()
