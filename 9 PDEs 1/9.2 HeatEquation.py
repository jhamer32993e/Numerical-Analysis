import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import animation, rc

# 9.2

t = np.linspace(0, 1, 101)
dt = t[1] - t[0]
x = np.linspace(0, 1, 21)
dx = x[1] - x[0]

k = 0.1
a = k * dt / dx**2

U = np.zeros((len(x), len(t)))
U[0, :] = 0
U[-1, :] = 0
U[:, 0] = np.sin(2 * np.pi * x)

for i in range(len(t) - 1):
    U[1:-1, i + 1] = U[1:-1, i] + a * (U[2:, i] - 2 * U[1:-1, i] + U[:-2, i])

plt.plot(x, U[:, -1])
plt.xlabel("x")
plt.ylabel("U")
plt.grid()
plt.show()

print(U[5, 20])

print("- " * 40)


# 9.3
def plotSolution(x, t, U):
    fig = go.Figure(data=[go.Surface(z=U.T, x=x, y=t)])
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u"),
    )
    return fig


def animateSolution(x, t, U):
    fig, ax = plt.subplots()
    plt.close()
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_xlim((np.min(x), np.max(x)))
    ax.set_ylim((np.min(U), np.max(U)))
    (frame,) = ax.plot([], [], linewidth=2)

    step = int(len(t) / 30) + 1
    frames = range(0, int(len(t) / step), 1)

    def animator(i):
        n = i * step
        ax.set_title(f"t = {t[n]:.2f}")
        frame.set_data(x, U[:, n])
        return (frame,)

    ani = animation.FuncAnimation(fig, animator, frames=frames, interval=100)
    rc("animation", html="jshtml")  # embed in the HTML for Google Colab
    return ani


# 9.4
def Heat1D(k, t0, tmax, NumT, x0, xmax, NumX, IC, LeftBC, RightBC):
    t = np.linspace(t0, tmax, NumT + 1)
    dt = t[1] - t[0]
    x = np.linspace(x0, xmax, NumX + 1)
    dx = x[1] - x[0]

    U = np.zeros((len(x), len(t)))
    U[0, :] = LeftBC(t)
    U[-1, :] = RightBC(t)

    U[:, 0] = IC(x)

    a = k * dt / dx**2
    for i in range(len(t) - 1):
        U[1:-1, i + 1] = U[1:-1, i] + a * (U[2:, i] - 2 * U[1:-1, i] + U[:-2, i])

    return x, t, U


x, t, U = Heat1D(
    0.1, 0, 1, 100, 0, 1, 20, lambda x: np.sin(2 * np.pi * x), lambda t: 0, lambda t: 0
)
plotSolution(x, t, U).show()
animateSolution(x, t, U)

print("- " * 40)

# 9.5
x, t, U = Heat1D(
    0.1, 0, 1, 100, 0, 1, 40, lambda x: np.sin(2 * np.pi * x), lambda t: 0, lambda t: 0
)
plotSolution(x, t, U).show()

print("- " * 40)

# 9.7
x, t, U = Heat1D(
    1, 0, 0.1, 100, 0, 1, 10, lambda x: np.sin(np.pi * x), lambda t: 0, lambda t: 0
)
plotSolution(x, t, U).show()

X, T = np.meshgrid(x, t, indexing="ij")
u_exact = np.exp(-np.pi**2 * T) * np.sin(np.pi * X)
plotSolution(x, t, u_exact).show()

print("- " * 40)

# 9.8
x, t, U = Heat1D(
    1,
    0,
    0.1,
    100,
    0,
    1,
    10,
    lambda x: np.sin(np.pi * x) + np.sin(3 * np.pi * x),
    lambda t: 0,
    lambda t: 0,
)
plotSolution(x, t, U).show()

X, T = np.meshgrid(x, t, indexing="ij")
u_exact = np.exp(-np.pi**2 * T) * np.sin(np.pi * X) + np.exp(
    -9 * np.pi**2 * T
) * np.sin(3 * np.pi * X)
plotSolution(x, t, u_exact).show()

print("- " * 40)

# 9.9
x, t, U = Heat1D(
    0.5,
    0,
    1,
    1000,
    0,
    1,
    10,
    lambda x: np.sin(2 * np.pi * x),
    lambda t: 0,
    lambda t: np.sin(5 * np.pi * t),
)
plotSolution(x, t, U).show()

print("- " * 40)


# 9.10
def MasterHeat1D(
    k, t0, tmax, NumT, x0, xmax, NumX, IC, LeftBC, LbcType, RightBC, RbcType
):
    t = np.linspace(t0, tmax, NumT + 1)
    dt = t[1] - t[0]
    x = np.linspace(x0, xmax, NumX + 1)
    dx = x[1] - x[0]

    U = np.zeros((len(x), len(t)))
    U[:, 0] = IC(x)

    a = k * dt / dx**2
    for i in range(len(t) - 1):
        U[1:-1, i + 1] = U[1:-1, i] + a * (U[2:, i] - 2 * U[1:-1, i] + U[:-2, i])
        if LbcType == "D":
            U[0, i + 1] = LeftBC(t[i + 1])

        elif LbcType == "N":
            q = LeftBC(t[i + 1])
            U[0, i + 1] = U[1, i + 1] - (q * dx)

        if RbcType == "D":
            U[-1, i + 1] = RightBC(t[i + 1])
        elif RbcType == "N":
            q = RightBC(t[i + 1])
            U[-1, i + 1] = U[-2, i + 1] + (q * dx)
    return x, t, U


x, t, U = MasterHeat1D(
    1,
    0,
    1,
    1000,
    0,
    1,
    10,
    lambda x: np.cos(np.pi * x / 2),
    lambda t: 0,
    "N",
    lambda t: 0,
    "D",
)
plotSolution(x, t, U).show()
