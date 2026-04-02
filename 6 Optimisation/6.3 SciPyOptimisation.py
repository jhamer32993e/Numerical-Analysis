import numpy as np
from scipy.optimize import minimize as minimise

# 6.16
f = lambda x: (x - 3) ** 2 - 5
print(minimise(f, 2))  # (function, x0)

print()

f = lambda x, y: np.sin(x) * np.exp(-np.sqrt(x**2 + y**2))
print(minimise(f, *[0, 0]))

# Outputs: success?, status 0 means success no errors, x is where the min was
# fun is the value at the min, jac is the gradient/jacobian at the min ~=0
# hess_inv is inverse of hessian matrix = second deriv, nit is no. iterations
# nfev is no. function evaluations, njev no. jacobian evaluations
