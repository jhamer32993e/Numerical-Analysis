import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as minimise

# 6.18
x = np.array([0, 1, 2, 3])
y = np.array([2.37, 4.14, 12.22, 23.68])
plt.plot(x, y, "bo", label="Data")

a = 1
b = 1
c = 0
x = np.linspace(0, 4, 100)
guess = a * x**2 + b * x + c
plt.plot(x, guess, "r--", label="Guess")
plt.grid()
plt.legend()
plt.show()

print("- " * 40)

# 6.19
xdata = np.array([0, 1, 2, 3])
ydata = np.array([2.37, 4.14, 12.22, 23.68])
b = 2
c = 0.75
A = np.linspace(1, 5, 100)
SumSqRes = []  # this is storage for the sum of the sq. residuals
for a in A:
    guess = a * xdata**2 + b * xdata + c
    residuals = guess - ydata
    SumSqRes.append(residuals**2)  # calculate the sum of the squ. residuals
plt.plot(A, SumSqRes)
plt.grid()
plt.xlabel("Value of a")
plt.ylabel("Sum of squared residuals")
plt.show()
# min error at a~=1.8

print("- " * 40)

# 6.20
xdata = np.array([0, 1, 2, 3])
ydata = np.array([2.37, 4.14, 12.22, 23.68])
parameters = np.array([1.8, 2, 0.75])


def SSRes(a):
    yapprox = a[0] * xdata**2 + a[1] * xdata + a[2]
    residuals = np.abs(ydata - yapprox)

    return np.sum(residuals**2)


BestParameters = minimise(SSRes, parameters)
print("The minimization diagnostics are: \n", BestParameters)

plt.plot(xdata, ydata, "bo")
x = np.linspace(0, 4, 100)
y = BestParameters.x[0] * x**2 + BestParameters.x[1] * x + BestParameters.x[2]
plt.plot(x, y, "r--")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Best Fit Quadratic")
plt.show()

print("- " * 40)

# 6.21
# Set a seed for the random number generator
np.random.seed(1)
# Use a quadratic function to generate some fake data
a, b, c = 2, 2, 0.75
f = lambda x: a * x**2 + b * x + c
# Choose 4 equally-spaced x-values
xdata = np.linspace(0, 3, 100)
# Add normally distributed errors to the y values
ydata = f(xdata) + np.random.normal(0, 1, 100)
# round to two digits
ydata = np.around(ydata, 2)

BestParameters = minimise(SSRes, parameters)
print("The minimization diagnostics are: \n", BestParameters)

plt.plot(xdata, ydata, "bo")
x = np.linspace(0, 4, 100)
y = BestParameters.x[0] * x**2 + BestParameters.x[1] * x + BestParameters.x[2]
plt.plot(x, y, "r--")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Best Fit Quadratic")
plt.show()
