import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
import plotly.graph_objects as go
import warnings

# 10.2
x = np.linspace(0, 1, 101)
y = x
dx = x[1] - x[0]
dy = dx
t = np.linspace(0, 0.1, 10001)
dt = t[1] - t[0]

X, Y = np.meshgrid(x, y, indexing="ij")
U = np.zeros((len(x), len(y), len(t)))
IC = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
U[:, :, 0] = IC(X, Y)
U[0, :, :] = 0
U[-1, :, :] = 0
U[:, 0, :] = 0
U[:, -1, :] = 0

k = 1
a = k * dt / dx**2

for n in range(len(t) - 1):
    U[1:-1, 1:-1, n + 1] = U[1:-1, 1:-1, n] + a * (
        U[2:, 1:-1, n]
        + U[1:-1, 2:, n]
        - 4 * U[1:-1, 1:-1, n]
        + U[:-2, 1:-1, n]
        + U[1:-1, :-2, n]
    )


def animate_solution_2d(t, x, y, U, skipFrames=50):
    Y, X = np.meshgrid(y, x)

    # Set up the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Initialize the surface plot
    surface = [ax.plot_surface(X, Y, U[:, :, 0], cmap="inferno")]

    # Don't display every time
    step = int(len(t) / 30) + 1
    frames = int(len(t) / step)

    def animate(i):
        n = i * step
        # Update the data of the surface plot for each frame
        ax.clear()  # Clear the previous frame
        surface[0] = ax.plot_surface(X, Y, U[:, :, n], cmap="inferno")
        ax.set_zlim(np.min(U), np.max(U))
        ax.set_title(f"Time: {t[n]:.2f}")

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=frames, repeat=True)
    plt.close()

    # Display the animation
    return HTML(ani.to_jshtml())


anim1 = animate_solution_2d(t, x, y, U)
display(anim1)

print("- " * 40)


# Heat Equation Solver
def MasterHeat2D(
    k,
    t0,
    tmax,
    NumT,
    x0,
    xmax,
    NumX,
    y0,
    ymax,
    NumY,
    IC,
    xLeftBC,
    xLbcType,
    xRightBC,
    xRbcType,
    yLeftBC,
    yLbcType,
    yRightBC,
    yRbcType,
):
    t = np.linspace(t0, tmax, NumT + 1)
    dt = t[1] - t[0]

    x = np.linspace(x0, xmax, NumX + 1)
    dx = x[1] - x[0]

    y = np.linspace(y0, ymax, NumY + 1)
    dy = y[1] - y[0]

    a_x = k * dt / dx**2
    a_y = k * dt / dy**2

    X, Y = np.meshgrid(x, y, indexing="ij")
    U = np.zeros((len(x), len(y), len(t)))

    # IC
    U[:, :, 0] = IC(X, Y)

    for i in range(len(t) - 1):
        U[1:-1, 1:-1, i + 1] = (
            U[1:-1, 1:-1, i]
            + a_x * (U[2:, 1:-1, i] - 2 * U[1:-1, 1:-1, i] + U[:-2, 1:-1, i])
            + a_y * (U[1:-1, 2:, i] - 2 * U[1:-1, 1:-1, i] + U[1:-1, :-2, i])
        )

        # X BC
        if xLbcType == "D":
            U[0, :, i + 1] = xLeftBC(t[i + 1])
        elif xLbcType == "N":
            q = xLeftBC(t[i + 1])
            U[0, :, i + 1] = U[1, :, i + 1] - (q * dx)

        if xRbcType == "D":
            U[-1, :, i + 1] = xRightBC(t[i + 1])
        elif xRbcType == "N":
            q = xRightBC(t[i + 1])
            U[-1, :, i + 1] = U[-2, :, i + 1] + (q * dx)

        # Y BC
        if yLbcType == "D":
            U[:, 0, i + 1] = yLeftBC(t[i + 1])
        elif yLbcType == "N":
            q = yLeftBC(t[i + 1])
            U[:, 0, i + 1] = U[:, 1, i + 1] - (q * dy)

        if yRbcType == "D":
            U[:, -1, i + 1] = yRightBC(t[i + 1])
        elif yRbcType == "N":
            q = yRightBC(t[i + 1])
            U[:, -1, i + 1] = U[:, -2, i + 1] + (q * dy)

    return x, y, t, U

    # x, y, t, U = MasterHeat2D(
    k = (1.0,)
    t0 = (0,)
    tmax = (0.1,)
    NumT = (5000,)
    x0 = (0,)
    xmax = (1.0,)
    NumX = (100,)
    y0 = (0,)
    ymax = (1.0,)
    NumY = (100,)
    IC = (IC,)
    xLeftBC = (lambda t: 0,)
    xLbcType = ("D",)
    xRightBC = (lambda t: 0,)
    xRbcType = ("D",)
    yLeftBC = (lambda t: 0,)
    yLbcType = ("D",)
    yRightBC = (lambda t: 0,)
    yRbcType = ("D",)


# )

# anim2 = animate_solution_2d(t, x, y, U)
# display(anim2)


# 10.6
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
    0.25,
    1000,
    0,
    1,
    20,
    lambda x: np.sin(np.pi * x),
    lambda t: 0,
    "D",
    lambda t: 0,
    "D",
)

exact = lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def animateComparison(x, t, U, exact_func):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("U(x, t)")
    ax.set_xlim((np.min(x), np.max(x)))

    # Pad the Y-axis slightly so the curves don't touch the very top/bottom
    ax.set_ylim((np.min(U) - 0.1, np.max(U) + 0.1))

    # Initialize TWO line objects (one for approximate, one for exact)
    (line_approx,) = ax.plot(
        [], [], "bo-", label="Approximate", markersize=4, alpha=0.7
    )
    (line_exact,) = ax.plot([], [], "r-", label="Exact", linewidth=2)

    ax.legend(loc="upper right")

    # Frame step logic
    step = int(len(t) / 30) + 1
    frames = range(0, int(len(t) / step), 1)

    def animator(i):
        n = i * step
        current_t = t[n]
        ax.set_title(f"1D Heat Equation - Time: {current_t:.4f}")

        # Update both lines for the current time step
        line_approx.set_data(x, U[:, n])
        line_exact.set_data(x, exact_func(x, current_t))

        return line_approx, line_exact

    # blit=True makes the animation render significantly faster
    ani = animation.FuncAnimation(fig, animator, frames=frames, interval=100, blit=True)

    plt.close(fig)  # Prevents a duplicate static plot from appearing

    return HTML(ani.to_jshtml())


anim = animateComparison(x, t, U, exact)
display(anim)

print("- " * 40)

# 10.7
x, y, t, U = MasterHeat2D(
    1,
    0,
    1,
    1000,
    0,
    1,
    200,
    0,
    1,
    200,
    lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
    lambda t: 0,
    "D",
    lambda t: 0,
    "D",
    lambda t: 0,
    "D",
    lambda t: 0,
    "D",
)

warnings.filterwarnings("ignore")
fig = go.Figure(data=[go.Surface(z=np.clip(U[:, :, 10].T, -2, 2), x=x, y=y)])
fig.update_layout(
    width=800,
    height=600,
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="u"),
)
