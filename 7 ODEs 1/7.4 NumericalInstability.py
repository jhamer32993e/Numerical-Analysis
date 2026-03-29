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


# 7.20
f = lambda x, t: -3 * x
x0 = 1
t0 = 0
tmax = 10
dt = [0.1, 0.5, 0.7, 1.0]
colours = ["g", "r", "m", "b"]
for i in range(len(dt)):
    print(i + 1)
    print("- " * 40)
    print(Euler1D(f, x0, t0, tmax, dt[i]))


for i in range(len(dt)):
    plt.plot(
        Euler1D(f, x0, t0, tmax, dt[i])[1],
        Euler1D(f, x0, t0, tmax, dt[i])[0],
        colours[i],
        label=f"h={dt[i]}",
    )

exact = lambda t: np.exp(-3 * t)
t = np.linspace(0, 10, 100)
plt.plot(t, exact(t), "y-", label="exact")
plt.ylim(-5, 5)
plt.grid()
plt.title("Euler's Method")
plt.legend()
plt.show()


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


for i in range(len(dt)):
    plt.plot(
        Midpoint1D(f, x0, t0, tmax, dt[i])[1],
        Midpoint1D(f, x0, t0, tmax, dt[i])[0],
        colours[i],
        label=f"h={dt[i]}",
    )

exact = lambda t: np.exp(-3 * t)
t = np.linspace(0, 10, 100)
plt.plot(t, exact(t), "y-", label="exact")
plt.ylim(-5, 5)
plt.grid()
plt.title("Midpoint Method")
plt.legend()
plt.show()
