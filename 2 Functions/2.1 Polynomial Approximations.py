import matplotlib.pyplot as plt
import numpy as np
import math

# 2.4, 2.5
x = np.arange(-1, 1, 0.01)
y = np.exp(x)
y0 = x / x
y1 = 1 + x
y2 = 1 + x + 0.5 * x**2
y3 = 1 + x + 0.5 * x**2 + 1 / 6 * x**3
exp = plt.plot(x, y, label="exp")
cub = plt.plot(x, y3, label="cubic")
quad = plt.plot(x, y2, label="quadratic")
lin = plt.plot(x, y1, label="linear")
const = plt.plot(x, y0, label="constant")
plt.legend()
plt.show()


# 2.6
def exp_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 1.0
    for i in range(1, n + 1):  # Start inclusive, stop exclusive
        approximation += x**i / (np.prod(range(1, i + 1)))
    return approximation


print(exp_approx(1, 40))
print("e^-1 =", exp_approx(-1, 40))
acc = np.abs((exp_approx(-1, 40) - np.exp(-1))) / np.exp(-1) * 100
print("Accuracy is:", acc, "%")


def my_exp_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 1.0
    added = x
    for i in range(1, n + 1):
        approximation += added
        added = added * x / (i + 1)
    return approximation


# 2.10
def sin_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 0
    for i in range(n):
        approximation += (x ** (2 * i + 1) * ((-1) ** i)) / math.factorial(2 * i + 1)
    return approximation


print(sin_approx(0.5, 10))
print(np.sin(0.5))

x = np.arange(-np.pi, np.pi, 0.01)
y1 = sin_approx(x, 1)
y2 = sin_approx(x, 2)
y3 = sin_approx(x, 3)
sin = np.sin(x)
t5 = plt.plot(x, y3, label="3 Terms")
t3 = plt.plot(x, y2, label="2 Term")
t1 = plt.plot(x, y1, label="1 Terms")
sinplot = plt.plot(x, sin, label="sinx")
plt.legend()
plt.show()


# 2.12
def log_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 0
    for i in range(1, n + 1):
        approximation += (((x - 1) ** i) * (-1) ** (i + 1)) / i
    return approximation


print(log_approx(0.9, 10))
print(np.log(0.9))

x = np.arange(0.1, 2, 0.01)
log = np.log(x)
y1 = log_approx(x, 1)
y2 = log_approx(x, 2)
y3 = log_approx(x, 3)
y4 = log_approx(x, 4)
t1 = plt.plot(x, y1, label="Linear")
t2 = plt.plot(x, y2, label="Quadratic")
t3 = plt.plot(x, y3, label="Cubic")
t4 = plt.plot(x, y4, label="Quartic")
logplot = plt.plot(x, log, label="Log")
plt.legend()
plt.show()
