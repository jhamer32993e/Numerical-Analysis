import numpy as np
import matplotlib.pyplot as plt


# 7.26
def Euler(F, x0, t0, tmax, dt):
    N = round((tmax - t0) / dt)
    dt = (tmax - t0) / N

    t = np.linspace(t0, tmax, N + 1)
    x = np.zeros((len(t), len(x0)))  # 2D array
    x[0, :] = x0
    for n in range(N):
        x[n + 1, :] = x[n, :] + dt * F(x[n, :], t[n])

    return x, t


b = 0.0003
g = 0.1
F = lambda x, t: np.array([-b * x[0] * x[1], b * x[0] * x[1] - g * x[1], g * x[1]])
x0 = [999, 1, 0]
t0 = 0
tmax = 100
dt = 0.01
x, t = Euler(F, x0, t0, tmax, dt)

plt.plot(t, x[:, 0], "g-", label="S")
plt.plot(t, x[:, 1], "r-", label="I")
plt.plot(t, x[:, 2], "b-", label="R")
plt.title("SIR Model")
plt.grid()
plt.legend()
plt.show()

print("- " * 40)


# 7.27
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


exact = lambda t: (2 / np.sqrt(3)) * np.exp(-t / 2) * np.sin((np.sqrt(3) / 2) * t)
F = lambda x, t: np.array([x[1], -x[1] - x[0]])
x0 = [0, 1]
t0 = 0
tmax = 10

DT = [10 ** (-(i + 1)) for i in range(4)]

EulerError = np.zeros(len(DT))
MPError = np.zeros(len(DT))

for i, dt in enumerate(DT):
    EulerX, EulerT = Euler(F, x0, t0, tmax, dt)
    EulerError[i] = np.max(np.abs(EulerX[:, 0] - exact(EulerT)))

    MPX, MPT = Midpoint(F, x0, t0, tmax, dt)
    MPError[i] = np.max(np.abs(MPX[:, 0] - exact(MPT)))

plt.loglog(DT, EulerError, "g-o", label="Euler Error")
plt.loglog(DT, MPError, "b-o", label="Midpoint Error")
plt.title("Error Convergence: Euler vs Midpoint")
plt.xlabel("Step Size (dt)")
plt.ylabel("Maximum Absolute Error")
plt.grid()
plt.legend()
plt.show()

print("- " * 40)

# 7.28
a = 0.2
b = 0.8
g = 0.1
x0 = [2, 0]
t0 = 0
tmax = 60
dt = 0.01
F = lambda x, t: np.array([-a * x[1], b * x[0] - g * (x[1] ** 2)])

x, t = Midpoint(F, x0, t0, tmax, dt)
print(f"Romeo: {x[-1, 0]}, Juliet: {x[-1, 1]}")

print()


def secant(f, x0, x1, tol=1e-10):
    xnm1 = x0
    xn = x1
    while np.abs(xn - xnm1) > tol:
        xnp1 = xn - f(xn) * (xn - xnm1) / (f(xn) - f(xnm1))
        xnm1 = xn
        xn = xnp1
    return xn


F = lambda x, t, g: np.array([-a * x[1], b * x[0] - g * (x[1] ** 2)])
tmax = 30
dt = 0.01
gamma = secant(
    lambda g: Midpoint(lambda x, t: F(x, t, g), x0, t0, tmax, dt)[0][-1, 1], 0.1, 0.5
)
print(f"Gamma value such that y(30)=0: {gamma}")

print("- " * 40)

# 7.30
F = lambda x, t: np.array(
    [-0.05 * x[0] + 0.02 / 500 * x[0] * x[1], 0.25 * x[1] - 0.1 / 150000 * x[0] * x[1]]
)
x0 = [75000, 150]
t0 = 0
tmax = 100
dt = 0.01

x, t = Midpoint(F, x0, t0, tmax, dt)

plt.plot(t, x[:, 0], "g-", label="Blue Whale Pop")
plt.plot(t, x[:, 1], "r-", label="Krill Density")
plt.title("Blue Whale Pop and Krill Density over time")
plt.grid()
plt.legend()
plt.show()

print()

# If we have whaling, introduce a negative constant term in the whale ODE
# for the meeting of a quota of dead whales, ie 3000 per year
FWhaling = lambda x, t: np.array(
    [
        -0.05 * x[0] + 0.02 / 500 * x[0] * x[1] - 3000,
        0.25 * x[1] - 0.1 / 150000 * x[0] * x[1],
    ]
)
x, t = Midpoint(FWhaling, x0, t0, tmax, dt)

plt.plot(t, x[:, 0], "g-", label="Blue Whale Pop")
plt.plot(t, x[:, 1], "r-", label="Krill Density")
plt.title("Blue Whale Pop and Krill Density over time, with Whaling")
plt.grid()
plt.legend()
plt.show()
