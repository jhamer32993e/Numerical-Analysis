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
