import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1.5, 100)
y = lambda x: -np.exp(-(x**2)) - np.sin(x**2)
plt.plot(x, y(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, which="both", ls="--")
plt.show()

y = y(x)
print(x[np.argmin(y)])
