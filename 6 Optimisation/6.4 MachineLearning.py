import numpy as np
import matplotlib.pyplot as plt

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
