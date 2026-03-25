import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4]
y = [6, 3, 1.5, 0.75, 0.375]
f = lambda x: 6 * np.exp(-x / 2)
plotx = np.linspace(0, 5, 100)
t_grid = np.linspace(0, max(x) + 0.5, 20)
x_grid = np.linspace(-1, 7, 20)
T, X = np.meshgrid(t_grid, x_grid)
U = np.ones_like(T)
V = -X / 2
N = np.sqrt(U**2 + V**2)
U = U / N
V = V / N

plt.figure(figsize=(10, 6))
plt.quiver(
    T,
    X,
    U,
    V,
    color="lightgray",
    angles="xy",
    headwidth=3,
    label="Slope Field (x' = -0.5x)",
)

plt.plot(plotx, f(plotx), "b-", linewidth=2, label="Exact Curve")
plt.plot(x, y, "ro--", linewidth=2, markersize=8, label="Your Euler Points")
plt.title("Euler's Method vs. Exact Solution", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("Value (x)", fontsize=12)
plt.axhline(0, color="black", linewidth=0.8)  # x-axis
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper right")
plt.xlim(0, max(x) + 0.5)
plt.ylim(-1, 7)
plt.show()

print("- " * 40)


# 7.5
def Euler1D(f, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros(len(t))

    x[0] = x0
    for n in range(N):
        x[n + 1] = x[n] + dt * f(x[n], t[n])

    return x, t


# 7.6
f = lambda x, t: -1 / 3 * x + np.sin(t)
xExact = lambda t: 0.1 * (19 * np.exp(-t / 3) + 3 * np.sin(t) - 9 * np.cos(t))
x0 = 1
t0 = 0
tmax = 10
dt = 0.5
x, t = Euler1D(f, x0, t0, tmax, dt)
plt.plot(t, x, "b-", label="Euler")

t_highres = np.linspace(t0, tmax, 100)
plt.plot(t_highres, xExact(t_highres), "r--", label="Exact")
plt.legend()
plt.grid()
plt.show()

print("- " * 40)


# 7.7
def EulerError(f, x0, t0, tmax, hFactor, xExact):
    dt = hFactor ** (-np.linspace(0, 4, 50))
    errors = np.zeros_like(dt)
    for i in range(len(dt)):
        xApprox, t = Euler1D(f, x0, t0, tmax, dt[i])
        errors[i] = np.max(np.abs(xExact(t) - xApprox))

    plt.loglog(dt, errors)
    plt.xlabel("Step size")
    plt.ylabel("Maximum error")
    plt.grid()
    plt.show()


EulerError(f, x0, t0, tmax, 10, xExact)
