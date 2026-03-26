import numpy as np
import matplotlib.pyplot as plt


# 7.9
def Euler(F, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros((len(t), len(x0)))  # 2D array
    x[0, :] = x0
    for n in range(N):
        x[n + 1, :] = x[n, :] + dt * F(x[n, :], t[n])

    return x, t


m = 2
b = 40
k = 128
F = lambda x, t: np.array([x[1], -(k / m) * x[0] - (b / m) * x[1]])
x0 = [0, 0.6]
t0 = 0
tmax = 5
dt = 0.01
x, t = Euler(F, x0, t0, tmax, dt)

plt.plot(t, x[:, 0], "b-", t, x[:, 1], "r--")
plt.grid()
plt.title("Time Evolution of Position and Velocity")
plt.legend(["Position", "Velocity"])
plt.xlabel("Time")
plt.ylabel("Position and Velocity")
plt.show()

plt.plot(x[:, 0], x[:, 1])
plt.grid()
plt.title("Phase Plot")
plt.ylabel("Velocity")
plt.xlabel("Position")
plt.show()

print("- " * 40)

# 7.10
a, b, c, d = 1, 0.1, 1, 0.1
F = lambda x, t: np.array([a * x[0] - b * x[0] * x[1], d * x[0] * x[1] - c * x[1]])
x0 = [10, 5]
t0 = 0
tmax = 20
dt = 0.0001
x, t = Euler(F, x0, t0, tmax, dt)

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
