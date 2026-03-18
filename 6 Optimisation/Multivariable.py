import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 2D plot example
f = lambda x, y: np.sin(x) * np.exp(-np.sqrt(x**2 + y**2))

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
plt.tight_layout()
plt.show()


# Gradient example
def df(x):
    # Calculate some values needed in the gradient
    s = np.sqrt(x[0] ** 2 + x[1] ** 2)
    e = np.exp(-s)

    # Need to handle case x=0,y=0 separately to avoid division by zero
    if s == 0:
        return np.array([1, 0])

    gradient = np.zeros(2)
    gradient[0] = (np.cos(x[0]) - np.sin(x[0]) * x[0] / s) * e
    gradient[1] = -np.sin(x[0]) * x[1] / s * e

    return gradient


# 6.14
def GradientDescent(grad, x0, alpha, tol=10e-12, MaxIterations=1000):
    x = x0

    for i in range(MaxIterations):
        gradient = grad(*x)  # have to unpack the array ~x
        xnew = x - alpha * gradient

        if np.linalg.norm(xnew - x) < tol:
            return xnew

        x = xnew

    raise ValueError("Does not converge")


grad0 = lambda x, y: np.array(
    [
        (np.cos(x) - (x * np.sin(x)) / np.sqrt(x**2 + y**2))
        * np.exp(-np.sqrt(x**2 + y**2)),
        -(y * np.sin(x) / np.sqrt(x**2 + y**2)) * np.exp(-np.sqrt(x**2 + y**2)),
    ]
)
x0 = [-1, 1]
alpha = 1
tol = 10e-6
print(GradientDescent(grad0, x0, alpha, tol))


# 6.15
def GradientDescentPath(grad, x0, alpha, tol=1e-12, MaxIterations=1000):
    x = x0
    steps = []
    for i in range(MaxIterations):
        gradient = grad(*x)
        xnew = x - alpha * gradient
        steps.append(xnew)
        if np.linalg.norm(xnew - x) < tol:
            return steps

        x = xnew

    raise ValueError("Does not converge")


grad1 = lambda x, y: np.array([2 * x, 200 * y])
x0 = [1, 1]
alpha = 0.009
path = GradientDescentPath(grad1, x0, alpha, tol=1e-6, MaxIterations=1000)
print(len(path))


def plot_path(func, path, x_range=(-2, 2), y_range=(-2, 2), grid_pts=100):
    """
    Plots an interactive 3D surface and overlays an optimization path.
    Assumes 'func' takes x and y as separate arguments: f(x, y).
    """
    # 1. Convert your returned list of steps into a 2D numpy array
    path = np.array(path)

    # 2. Define the grid
    x = np.linspace(x_range[0], x_range[1], grid_pts)
    y = np.linspace(y_range[0], y_range[1], grid_pts)
    X, Y = np.meshgrid(x, y)

    # 3. Evaluate the function over the grid directly
    Z = func(X, Y)

    # 4. Calculate z-coordinates for the path directly
    path_z = func(path[:, 0], path[:, 1])

    fig = go.Figure()

    # Surface plot
    fig.add_trace(
        go.Surface(z=Z, x=x, y=y, colorscale="Viridis", opacity=0.8, name="Surface")
    )

    # Optimization path
    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path_z,
            mode="lines+markers",
            marker=dict(size=3, color="red"),
            line=dict(color="red", width=3),
            name="Iterates",
        )
    )

    # End point marker
    fig.add_trace(
        go.Scatter3d(
            x=[path[-1, 0]],
            y=[path[-1, 1]],
            z=[path_z[-1]],
            mode="markers",
            marker=dict(size=5, color="white", line=dict(color="black", width=1)),
            name="End Point",
        )
    )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)"),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig.show()


f1 = lambda x, y: x**2 + 100 * y**2
grad1 = lambda x, y: np.array([2 * x, 200 * y])
alpha = 0.009  # Try different values
path = GradientDescentPath(grad1, [1.0, 1.0], alpha, tol=1e-6, MaxIterations=10000)
print(len(path))
plot_path(f1, path)

f2 = lambda x, y: (1 - x) ** 2 + 100 * (y - x**2) ** 2
grad2 = lambda x, y: np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])

alpha = 0.002  # Try different values
path = GradientDescentPath(grad2, [-0.1, 4.0], alpha, tol=1e-6, MaxIterations=10000)
print(len(path))
plot_path(f2, path, x_range=(-2, 2), y_range=(-2, 5))
