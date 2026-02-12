import numpy as np
import math as m

# 1.19
print(np.sqrt(5) ** 2)
print(np.sqrt(5) ** 2 == 5)

print(49 * (1 / 49))
print(49 * (1 / 49) == 1)

print(np.exp(np.log(3)))
print(np.exp(np.log(3)) == 3)

print(np.cbrt(7) ** 3)
print(np.cbrt(7) ** 3 == 7)

# 1.21
x = 0.0001
print(2 * (m.sin(x / 2)) ** 2)
print(1 - m.cos(x))
# cos(x) likely to be very close to 1 so this can cause loss of significant digits, so we should use the LHS

# 1.22
a = 1
b = 1000000
c = 1
x = (-b + m.sqrt(b**2 - 4 * a * c)) / (2 * a)
print("x =", x)
y = (2 * c) / (-b - m.sqrt(b**2 - 4 * a * c))
print("y=", y)
