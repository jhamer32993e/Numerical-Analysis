import numpy as np

# 1.19
print(np.sqrt(5) ** 2)
print(f"Does sqrt(5)^2 = 5? {np.sqrt(5) ** 2 == 5}")
print()

print(49 * (1 / 49))
print(f"Does 49*1/49 = 1? {49 * (1 / 49) == 1}")
print()

print(np.exp(np.log(3)))
print(f"Does e^log3 = 3? {np.exp(np.log(3)) == 3}")
print()

print(np.cbrt(7) ** 3)
print(f"Does cbrt(7)^3 = 7? {np.cbrt(7) ** 3 == 7}")

print("- " * 40)

# 1.21
x = 0.0001
print(2 * (np.sin(x / 2)) ** 2)
print(1 - np.cos(x))
# cos(x) likely to be very close to 1 so this can cause loss of significant digits, so we should use the LHS

print("- " * 40)

# 1.22
a = 1
b = 1000000
c = 1
x = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
print("x =", x)
y = (2 * c) / (-b - np.sqrt(b**2 - 4 * a * c))
# Alternative quadratic formula to avoid subtracting similar values
print("y=", y)
