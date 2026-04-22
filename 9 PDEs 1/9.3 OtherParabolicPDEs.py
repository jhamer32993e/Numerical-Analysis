import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import animation, rc
from IPython.display import display, HTML


# 9.11
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


def FisherKPP(k, t0, tmax, NumT, x0, xmax, NumX, IC, LeftBC, RightBC):
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
        U[1:-1, i + 1] = (
            U[1:-1, i]
            + a * (U[2:, i] - 2 * U[1:-1, i] + U[:-2, i])
            + dt * (U[1:-1, i] * (1 - U[1:-1, i]))
        )

    return x, t, U


x, t, U = FisherKPP(
    1,
    0,
    10,
    1000,
    0,
    50,
    100,
    lambda x: (1 + np.tanh((x - 40) / 2)) / 2,
    lambda t: 0,
    lambda t: 1,
)
anim1 = animateSolution(x, t, U)
display(HTML(anim1.to_jshtml()))

print("- " * 40)


# 9.12
def AdvectiveDiffusion(k, t0, tmax, NumT, x0, xmax, NumX, IC, LeftBC, RightBC):
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
        U[1:-1, i + 1] = (
            U[1:-1, i]
            + a * (U[2:, i] - 2 * U[1:-1, i] + U[:-2, i])
            - dt * (U[2:, i] - U[1:-1, i]) / dx
        )

    return x, t, U


def plotSolution(x, t, U):
    fig = go.Figure(data=[go.Surface(z=U.T, x=x, y=t)])
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u"),
    )
    return fig


x, t, U = AdvectiveDiffusion(
    k=0.1,
    t0=0,
    tmax=1,
    NumT=100,
    x0=0,
    xmax=1,
    NumX=20,
    IC=lambda x: np.sin(np.pi * x),
    LeftBC=lambda t: 0,
    RightBC=lambda t: 0,
)

plotSolution(x, t, U).show()
anim2 = animateSolution(x, t, U)
display(HTML(anim2.to_jshtml()))
