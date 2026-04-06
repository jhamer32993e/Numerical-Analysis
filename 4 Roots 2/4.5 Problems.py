import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# 4.26
def newton_with_error_tracking(f, fdash, x_exact, x0, tol=1e-10):
    errors = []
    x = x0
    xnew = x - (f(x) / fdash(x))
    while np.abs(x - xnew) > tol:
        if fdash(x) == 0:
            return ValueError("Derivative equals zero")
        x = xnew
        xnew = x - (f(x) / fdash(x))
        error = np.abs(xnew - x_exact)
        errors.append(error)
    return errors


def plot_errors(errors):
    # Creating the x values for the plot (iterations)
    iterations = np.arange(len(errors))

    # Plotting the errors
    plt.scatter(iterations, errors, label="Error per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error in Each Iteration - Bisection Method")
    plt.legend()
    plt.show()


def plot_log_errors(errors):
    # Convert errors to base 2 logarithm
    log_errors = np.log2(errors)
    # Creating the x values for the plot (iterations)
    iterations = np.arange(len(log_errors))

    # Plotting the errors
    plt.scatter(iterations, log_errors, label="Log Error per Iteration")

    # Determine slope and intercept of the best-fit straight line
    slope, intercept = np.polyfit(iterations, log_errors, deg=1)
    best_fit_line = slope * iterations + intercept
    # Plot the best-fit line
    plt.plot(iterations, best_fit_line, label="Best Fit Line", color="red")

    plt.xlabel("Iteration")
    plt.ylabel("Base 2 Log of Absolute Error")
    plt.title("Log Absolute Error in Each Iteration")
    plt.legend()
    plt.show()


f = lambda x: x**3 - 3
fdash = lambda x: 3 * x**2
xExact = np.cbrt(3)
x0 = 1
plot_errors(newton_with_error_tracking(f, fdash, xExact, x0, 1e-6))

plot_log_errors(newton_with_error_tracking(f, fdash, xExact, x0, 1e-6))

print("- " * 40)


# 4.27
def bisection(f, a, b, tol=1e-10):
    if f(a) * f(b) >= 0:
        raise ValueError(
            "f(a), f(b) do not have opposite signs, no guarantee of a root"
        )
    midpoint = (a + b) / 2
    while abs(a - b) > 2 * tol:
        if f(a) * f(midpoint) < 0:
            b = midpoint
        elif f(b) * f(midpoint) < 0:
            a = midpoint
        elif f(midpoint) == 0:
            return midpoint
        midpoint = (a + b) / 2
    return midpoint


def newton(f, fdash, x0, tol=1e-10):
    x = x0
    xnew = x - f(x) / fdash(x)
    for i in range(30):
        if fdash(x) == 0:
            return ValueError("Derivative goes to zero")
        x = xnew
        xnew = x - (f(x) / fdash(x))
        if np.abs(x - xnew) > tol:
            return x
    return "Does not converge"


def secant(f, x0, x1, tol=1e-10):
    xnm1 = x0
    xn = x1
    while np.abs(xn - xnm1) > tol:
        xnp1 = xn - f(xn) * (xn - xnm1) / (f(xn) - f(xnm1))
        xnm1 = xn
        xn = xnp1
    return xn


f = lambda x: 3 * np.sin(x) + 9 - x**2 + np.cos(x)
fdash = lambda x: 3 * np.cos(x) - 2 * x - np.sin(x)


def comparison(f, fdash, a, b, x0, x1):
    print("Bisection: ", bisection(f, a, b))
    print("Newton: ", newton(f, fdash, x0))
    print("Secant: ", secant(f, x0, x1))
    return


comparison(f, fdash, 2, 4, 3, 4)

print("- " * 40)


# 4.29
def SecondOrderTaylorSolve(f, fdash, fddash, x0, tol=1e-10):
    xp = x0
    # Positive root
    while abs(f(xp)) > tol:
        a = 0.5 * fddash(xp)
        b = fdash(xp) - xp * fddash(xp)
        c = f(xp) - fdash(xp) * xp + 0.5 * fddash(xp) * (xp**2)
        xp = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return xp


f = lambda x: x**3 - 3
fdash = lambda x: 3 * x**2
fddash = lambda x: 6 * x
print(SecondOrderTaylorSolve(f, fdash, fddash, 1))


def Taylor_with_error_tracking(f, fdash, fddash, x_exact, x0, tol=1e-10):
    errors = []
    xp = x0
    # Positive root
    while abs(f(xp)) > tol:
        a = 0.5 * fddash(xp)
        b = fdash(xp) - xp * fddash(xp)
        c = f(xp) - fdash(xp) * xp + 0.5 * fddash(xp) * (xp**2)
        xp = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        error = abs(xp - x_exact)
        errors.append(error)
    return errors


def plot_error_progression(errors):
    errors = [i for i in errors if i != 0]
    # Calculating the log2 of the absolute error at step n and n+1
    log_errors = np.log2(errors)
    log_errors_n = log_errors[:-1]  # log errors at step n (excluding the last one)
    log_errors_n_plus_1 = log_errors[
        1:
    ]  # log errors at step n+1 (excluding the first one)

    # Plotting log_errors_n+1 vs log_errors_n
    plt.scatter(
        log_errors_n, log_errors_n_plus_1, label="Log Error at n+1 vs Log Error at n"
    )

    # Fitting a straight line to the data points
    slope, intercept = np.polyfit(log_errors_n, log_errors_n_plus_1, deg=1)
    best_fit_line = slope * log_errors_n + intercept
    plt.plot(log_errors_n, best_fit_line, color="red", label="Best Fit Line")

    # Setting up the plot
    plt.xlabel("Log2 of Absolute Error at Step n")
    plt.ylabel("Log2 of Absolute Error at Step n+1")
    plt.title("Log2 of Absolute Error at Step n+1 vs Step n")
    plt.legend()
    plt.show()


plot_error_progression(Taylor_with_error_tracking(f, fdash, fddash, np.cbrt(3), 1))

print("- " * 40)

# 4.31
f = lambda x: x**3 * (x - 3) * (x - 6) ** 4
fdash = (
    lambda x: 8 * x**7
    - 189 * x**6
    + 1728 * x**5
    - 7560 * x**4
    + 15552 * x**3
    - 11664 * x**2
)
print(newton(f, fdash, 0.1))
# Doesnt do massively well because the slope is so steep
print()
print(newton(f, fdash, 5.9, 1e-15))
print("- " * 40)


# 4.32
def f(x):
    return x * np.sin(x) - np.log(x)


x = np.linspace(0, 5, 100)
plt.plot(x, f(x))
plt.grid()
plt.show()


print(fsolve(f, 3, full_output=1))
#
print()


def F(x):
    return [x[0] * np.cos(x[1]) - 4, x[0] * x[1] - x[1] - 5]


print(fsolve(F, [6, 1], full_output=1))
# Note: full_output=1 gives the solver diagnostics
print()


def F(x):
    return [x[0] ** 2 - x[0] * x[1] ** 2 - 2, x[0] * x[1] - 2]


print(fsolve(F, [1, 1], full_output=1))
