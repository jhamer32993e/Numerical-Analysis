import numpy as np
import matplotlib.pyplot as plt


# 3.10
def bisection(f, a, b, tol=1e-5):
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


# 3.11
f1 = lambda x: x**2 - 2
print("x =", bisection(f1, 0, 2))
print()

f2 = lambda x: np.sin(x) + x**2 - 2 * np.log(x) - 5
print("x =", bisection(f2, 1, 5))
print()

f3 = lambda x: 3 * np.sin(x) + 9 - x**2 - np.cos(x)
print("x =", bisection(f3, 1, 5, 1e-6))
print()

print("- " * 40)


# 3.14
def for_bisection(f, a, b, n):
    if f(a) * f(b) >= 0:
        raise ValueError(
            "f(a), f(b) do not have opposite signs, no guarantee of a root"
        )
    midpoint = (a + b) / 2
    for i in range(n):
        if f(a) * f(midpoint) < 0:
            b = midpoint
        elif f(b) * f(midpoint) < 0:
            a = midpoint
        elif f(midpoint) == 0:
            return midpoint
        midpoint = (a + b) / 2
    return midpoint


print("x =", for_bisection(f3, 1, 5, 22))


# Example 3.1
def bisection_with_error_tracking(f, x_exact, a, b, tol):
    errors = []
    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            break
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        error = abs(midpoint - x_exact)
        errors.append(error)
    return errors


f4 = lambda x: x**2 - 2
print(bisection_with_error_tracking(f4, np.sqrt(2), 1, 2, 1e-7))


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


plot_errors(bisection_with_error_tracking(f4, np.sqrt(2), 1, 2, 1e-7))


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


plot_log_errors(bisection_with_error_tracking(f4, np.sqrt(2), 1, 2, 1e-7))

print("- " * 40)

# 3.15
f5 = lambda x: np.exp(x - 3) + np.sqrt(x + 6) - 4
plot_errors(bisection_with_error_tracking(f5, 3, 0, 5, 1e-7))
plot_log_errors(bisection_with_error_tracking(f5, 3, 0, 5, 1e-7))


# Example 3.2
def plot_error_progression(errors):
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


# 3.16
plot_error_progression(bisection_with_error_tracking(f5, 3, 0, 5, 1e-7))

print("- " * 40)

f6 = lambda x: x**3 - 3
plot_errors(bisection_with_error_tracking(f6, np.cbrt(3), 1, 2, 1e-7))
plot_log_errors(bisection_with_error_tracking(f6, np.cbrt(3), 1, 2, 1e-7))
plot_error_progression(bisection_with_error_tracking(f6, np.cbrt(3), 1, 2, 1e-7))
