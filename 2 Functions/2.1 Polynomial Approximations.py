import matplotlib.pyplot as plt
import numpy as np
import math

# 2.4, 2.5
x = np.linspace(-1, 1, 200)
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
plt.title("Exponential Taylor Approximations")
plt.legend()
plt.show()

print("- " * 40)


# 2.6
def exp_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 1.0
    for i in range(1, n + 1):  # Start inclusive, stop exclusive
        approximation += x**i / (np.prod(range(1, i + 1)))
    return approximation


print(f"e ~= {exp_approx(1, 40)}")
acc = np.abs((exp_approx(1, 40) - np.exp(1))) / np.exp(1) * 100
print("e accuracy is:", acc, "%")

print("- " * 40)

# 2.7
print("e^-1 =", exp_approx(-1, 40))
acc = np.abs((exp_approx(-1, 40) - np.exp(-1))) / np.exp(-1) * 100
print("e^-1 accuracy is:", acc, "%")

print("- " * 40)


# 2.8
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
    for i in range((n + 1) // 2):
        approximation += (x ** (2 * i + 1) * ((-1) ** i)) / math.factorial(2 * i + 1)
    return approximation


print(f"My sin(0.5) approximation: {sin_approx(0.5, 5)}")
print("Numpy sin(0.5) value:", np.sin(0.5))


x = np.linspace(-np.pi, np.pi, 100)
y1 = sin_approx(x, 1)
y2 = sin_approx(x, 2)
y3 = sin_approx(x, 3)
sin = np.sin(x)

plt.plot(x, y3, label="3 Terms")
plt.plot(x, y2, label="2 Term")
plt.plot(x, y1, label="1 Terms")
plt.plot(x, sin, label="sinx")
plt.title("Sin Taylor Approximations")
plt.legend()
plt.show()

print("- " * 40)


# 2.12
def log_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 0
    for i in range(1, n + 1):
        approximation += (((x - 1) ** i) * (-1) ** (i + 1)) / i
    return approximation


print(f"My log(0.9) approximation: {log_approx(0.9, 10)}")
print("Numpy log(0.9) value:", np.log(0.9))

x = np.linspace(0.1, 2, 100)
log = np.log(x)
y1 = log_approx(x, 1)
y2 = log_approx(x, 2)
y3 = log_approx(x, 3)
y4 = log_approx(x, 4)

plt.plot(x, y1, label="Linear")
plt.plot(x, y2, label="Quadratic")
plt.plot(x, y3, label="Cubic")
plt.plot(x, y4, label="Quartic")
plt.plot(x, log, label="Log")
plt.title("Log Taylor Approximation")
plt.legend()
plt.show()
