import matplotlib.pyplot as plt
import numpy as np

# 6.2
x = np.linspace(0, 1.5, 1000)
y = lambda x: -np.exp(-(x**2)) - np.sin(x**2)
plt.plot(x, y(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, which="both", ls="--")
plt.show()

y = y(x)
print(x[np.argmin(y)])

print("- " * 40)


# 6.5
def GoldenSection(f, a, c, b, tol=1e-12):
    if a >= c or c >= b:
        raise ValueError("points not in order a < c < b")

    if f(a) <= f(c) or f(b) <= f(c):
        raise ValueError("f(c) not less than both f(a) and f(b)")
    rho = (3 - 5**0.5) / 2
    fc = f(c)
    while b - a >= tol:
        if b - c > c - a:
            x = c + rho * (b - c)
            fx = f(x)

            if fx < fc:
                a = c
                c = x
                fc = fx
            else:
                b = x

        else:
            x = c - rho * (c - a)
            fx = f(x)
            if fx < fc:
                b = c
                c = x
                fc = fx
            else:
                a = x

    return (a + b) / 2


f1 = lambda x: -np.exp(-(x**2)) - np.sin(x**2)
print(GoldenSection(f1, 0, 1, 2))

print("- " * 40)


# 6.9
def GradientDescent(fDash, x0, alpha, tol=1e-12, MaxIterations=1000):
    x = x0
    xnew = x - alpha * fDash(x)
    count = 0
    for i in range(MaxIterations):
        x = xnew
        grad = fDash(x)
        xnew = x - alpha * grad
        count += 1
        if np.abs(xnew - x) < tol:
            return xnew, count
    raise ValueError("Does not converge")


# 6.10
f2Dash = lambda x: 4 * np.cos(4 * x) * (x**2 - 10 * x) + (np.sin(4 * x) + 1) * (
    2 * x - 10
)
print(GradientDescent(f2Dash, 3, 0.003))

print("- " * 40)


# 6.12
def gradient_descent_with_error_tracking(fDash, alpha, x_exact, x0, tol=1e-12):
    errors = []
    x = x0
    errors.append(np.abs(x - x_exact))
    xnew = x - alpha * fDash(x)

    while np.abs(xnew - x) > tol:
        x = xnew
        errors.append(np.abs(x - x_exact))

        xnew = x - alpha * fDash(x)
        if fDash(x) == 0:
            break

    return errors


f3Dash = lambda x: -np.sin(x)
print(gradient_descent_with_error_tracking(f3Dash, 0.1, np.pi, 3))


def plot_error_progression(errors):
    # Calculating the log2 of the absolute error at step n and n+1
    log_errors = np.log2(errors)
    log_errors_n = log_errors[:-1]  # log errors at step n (excluding the last one)
    log_errors_n_plus_1 = log_errors[1:]
    # log errors at step n+1 (excluding the first one)

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


plot_error_progression(gradient_descent_with_error_tracking(f3Dash, 0.1, np.pi, 3))
