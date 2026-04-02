import numpy as np
import math


# 2.13 + 2.14 (final column)
def exp_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 1.0
    for i in range(1, n + 1):  # Start inclusive, stop exclusive
        approximation += x**i / (np.prod(range(1, i + 1)))
    return approximation


print("e^x Approximation and Truncation Error")
# Create table header
print(
    f"{'Order n':<8} | {'f_n(0.1)':<15} | {'ε_n(0.1)':<15} | "
    f"{'f_n(0.2)':<15} | {'ε_n(0.2)':<15} | {'ε_n(0.2) / ε_n(0.1)':<15} "
)
print("-" * 110)

# Fill in n=0 row
f_n = lambda x: 1
e_n = lambda x: abs(np.exp(x) - f_n(x))
print(
    f"{0:<8} | {f_n(0.1):<15.10g} | {e_n(0.1):<15.10g} | "
    f"{f_n(0.2):<15.10g} | {e_n(0.2):<15.10g} | {e_n(0.2)/e_n(0.1):<15.10g}"
)

# Fill in n=1 row
f_n = lambda x: 1 + x
e_n = lambda x: abs(np.exp(x) - f_n(x))
print(
    f"{1:<8} | {f_n(0.1):<15.10g} | {e_n(0.1):<15.10g} | "
    f"{f_n(0.2):<15.10g} | {e_n(0.2):<15.10g} | {e_n(0.2)/e_n(0.1):<15.10g}"
)

# Fill in more rows.
for n in range(2, 6):
    f_n = lambda x: exp_approx(x, n)
    e_n = lambda x: abs(np.exp(x) - f_n(x))
    print(
        f"{n:<8} | {f_n(0.1):<15.10g} | {e_n(0.1):<15.10g} | {f_n(0.2):<15.10g} | {e_n(0.2):<15.10g} | {e_n(0.2)/e_n(0.1):<15.10g}"
    )
print()
print("- " * 40)
print()


# 2.15
def sin_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 0
    for i in range(n):
        approximation += (x ** (2 * i + 1) * ((-1) ** i)) / math.factorial(2 * i + 1)
    return approximation


print("sin(x) Approximation and Truncation Error")
# Create table header
print(
    f"{'Order n':<8} | {'f_n(0.1)':<15} | {'ε_n(0.1)':<15} | "
    f"{'f_n(0.2)':<15} | {'ε_n(0.2)':<15} | {'ε_n(0.2) / ε_n(0.1)':<15} "
)
print("-" * 110)

# Fill in rows
for n in range(1, 6):
    f_n = lambda x: sin_approx(x, n)
    e_n = lambda x: abs(np.sin(x) - f_n(x))
    print(
        f"{2*n-1:<8} | {f_n(0.1):<15.10g} | {e_n(0.1):<15.10g} | {f_n(0.2):<15.10g} | {e_n(0.2):<15.10g} | {e_n(0.2)/e_n(0.1):<15.10g}"
    )
print()
print("- " * 40)
print()


# 2.16
def log_approx(x, n):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 0
    for i in range(1, n + 1):
        approximation += (((x - 1) ** i) * (-1) ** (i + 1)) / i
    return approximation


print("log(x) Approximation and Truncation Error")
# Create table header
print(
    f"{'Order n':<8} | {'f_n(1.02)':<15} | {'ε_n(1.02)':<15} | "
    f"{'f_n(1.1)':<15} | {'ε_n(1.1)':<15} | {'ε_n(1.1) / ε_n(1.02)':<15} "
)
print("-" * 110)

# Fill in rows
for n in range(1, 6):
    f_n = lambda x: log_approx(x, n)
    e_n = lambda x: abs(np.log(x) - f_n(x))
    print(
        f"{n:<8} | {f_n(1.02):<15.10g} | {e_n(1.02):<15.10g} | {f_n(1.1):<15.10g}"
        f" | {e_n(1.1):<15.10g} | {e_n(1.1)/e_n(1.02):<15.10g}"
    )
