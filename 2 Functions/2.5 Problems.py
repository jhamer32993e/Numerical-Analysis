import numpy as np


# 2.31
# Pi approximation using Taylor series of 1/(1+x), then x -> x^2, then integrate
# the resulting function = arctan(x), then pi = 4arctan(1)
def pi_approx(n, x=1):
    if n < 0:
        raise ValueError("n must be at least 0")
    approximation = 0
    for i in range((n + 1) // 2):
        approximation += x ** (2 * i + 1) * (-1) ** i / (2 * i + 1)
    approximation *= 4
    return approximation


print(pi_approx(10000000))
