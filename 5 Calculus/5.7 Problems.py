import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 5.48
def FirstDerivFromTable(data):
    array = np.array(data)
    x = array[:, 0]
    y = array[:, 1]
    dy = np.diff(y)  # difference between successive points in y
    dx = np.diff(x)
    yDash = dy / dx

    return list(zip(x[:-1], yDash))


# zip combines lists into 2D array ish
# list forces the zip output to be accessable as a list

x = np.linspace(0, 10, 101)
y = x**2
dataFrameOut = pd.DataFrame({"x": x, "y": y})
dataFrameOut.to_csv("test.csv", index=False)

dataFrameIn = pd.read_csv("test.csv")
dataPoints = list(zip(dataFrameIn["x"], dataFrameIn["y"]))

derivPoints = FirstDerivFromTable(dataPoints)
xDash, yDash = zip(*derivPoints)  # zip(*) unzips the array into separate arrays

plt.figure(figsize=(10, 8))

# Top subplot: Original function
plt.subplot(2, 1, 1)
plt.plot(x, y, "bo-", label="Original Data ($y = x^2$)")
plt.title("Underlying Function")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# Bottom subplot: Derivative approximation vs. Analytical exact
plt.subplot(2, 1, 2)
# Plot the numerical approximation
plt.plot(xDash, yDash, "ro", label="Numerical Approx (Forward Difference)")
# Plot the exact analytical derivative for comparison
plt.plot(x, 2 * x, "k--", label="Exact Analytical ($y' = 2x$)", alpha=0.6)
plt.title("First Derivative Approximation")
plt.xlabel("x")
plt.ylabel("y'")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# 5.49
def SecondDerivFromTable(data):
    array = np.array(data)
    x = array[:, 0]
    y = array[:, 1]
    h = x[1] - x[0]
    yDDash = (y[2:] - 2 * y[1:-1] + y[:-2]) / (h**2)

    return list(zip(x[1:-1], yDDash))


SecDerivPoints = SecondDerivFromTable(dataPoints)
xDDash, yDDash = zip(*SecDerivPoints)

plt.figure(figsize=(10, 8))

# Top subplot: Original function
plt.subplot(2, 1, 1)
plt.plot(x, y, "bo-", label="Original Data ($y = x^2$)")
plt.title("Underlying Function")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# Bottom subplot: Derivative approximation vs. Analytical exact
plt.subplot(2, 1, 2)
# Plot the numerical approximation
plt.plot(xDDash, yDDash, "ro", label="Numerical Approx (Forward Difference)")
# Plot the exact analytical derivative for comparison
plt.plot(x, np.full_like(x, 2), "k--", label="Exact Analytical ($y' = 2$)", alpha=0.6)
plt.title("Second Derivative Approximation")
plt.xlabel("x")
plt.ylabel("y''")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.ylim(0, 4)
plt.show()
