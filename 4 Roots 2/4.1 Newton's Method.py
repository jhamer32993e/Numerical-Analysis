import numpy as np
import matplotlib.pyplot as plt

# 4.6
x0 = 1
f = lambda x: x**2 - 2
fdash = lambda x: 2 * x


def table(f, fdash, x0):
    for i in range(5):
        print(
            f"{i:<2} | x={x0:<25.10g} | f(x)={f(x0):<30.10g} | f'(x)={fdash(x0):<25.10g}"
        )
        x0 = x0 - f(x0) / fdash(x0)


# 4.7
def newton(f, fdash, x0, tol=1e-10, MaxIterations=30):
    x = x0
    xnew = x - f(x) / fdash(x)
    for i in range(MaxIterations):
        if fdash(x) == 0:
            return ValueError("Derivative goes to zero")
        x = xnew
        xnew = x - (f(x) / fdash(x))
        if np.abs(x - xnew) > tol:
            return x
    return "Does not converge"


# print(newton(f, fdash, 1))

f1 = lambda x: x ** (1 / 3)
f1dash = lambda x: (1 / 3) * x ** (-2 / 3)
# table(f1, f1dash, 7)


# 4.14
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


print(newton_with_error_tracking(f, fdash, np.sqrt(2), 1))


# 4.15
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
    print(slope, intercept)
    best_fit_line = slope * log_errors_n + intercept
    plt.plot(log_errors_n, best_fit_line, color="red", label="Best Fit Line")

    # Setting up the plot
    plt.xlabel("Log2 of Absolute Error at Step n")
    plt.ylabel("Log2 of Absolute Error at Step n+1")
    plt.title("Log2 of Absolute Error at Step n+1 vs Step n")
    plt.legend()
    plt.show()


# plot_error_progression(newton_with_error_tracking(f, fdash, np.sqrt(2), 1))


# 4.16
# Edited prev function to print the formula of the polynomial it plots

# 4.17
f2 = lambda x: np.exp(x - 3) + np.sqrt(x + 6) - 4
f2dash = lambda x: np.exp(x - 3) + 0.5 * (x + 6) ** (-0.5)
# plot_error_progression(newton_with_error_tracking(f2, f2dash, 3, 1))
