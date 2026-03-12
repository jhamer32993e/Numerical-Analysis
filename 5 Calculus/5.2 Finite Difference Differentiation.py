import numpy as np
import matplotlib.pyplot as plt


# 5.11
def table(f, exact, delta_f):
    # Create table header
    hdr1 = "h"
    hdr2 = "Δf(1)"
    hdr3 = "|f'(1) - Δf(1)|"
    hdr4 = "Error reduction factor"
    print(f"{hdr1:<20} | {hdr2:<18} | {hdr3:<18} | {hdr4:<18}")
    print("-" * 82)

    # Fill in n=1 row
    h = 2 ** (-1)
    h_str = f"2^-1 = {h:g}"
    error = abs(exact - delta_f(h))
    print(f"{h_str:<20} | {delta_f(h):<18.15f} | {error:<18.15f} |")

    # Fill in n=2 row
    h = 2 ** (-2)

    h_str = f"2^-2 = {h:g}"
    error_prev = error
    error = abs(exact - delta_f(h))
    print(
        f"{h_str:<20} | {delta_f(h):<18.15f} | {error:<18.15f} | {error_prev/error:<18.15f}"
    )

    for i in range(3, 11):
        h = 2 ** (-i)
        h_str = f"2^{-i} = {h:g}"
        error_prev = error
        error = abs(exact - delta_f(h))
        print(
            f"{h_str:<20} | {delta_f(h):<18.15f} | {error:<18.15f} | {error_prev/error:<18.15f}"
        )
    print()
    return


f = lambda x: np.sin(x) * (1 - x)
exact = -np.sin(1)  # Derivative evaluated at 1
delta_f = lambda h: (f(1 + h) - f(1)) / h

table(f, exact, delta_f)
f1 = lambda x: np.sin(x)
exact1 = np.cos(1)
delta_f1 = lambda h: (f1(1 + h) - f1(1)) / h
table(f1, exact1, delta_f1)


# exact is the value of the deriv evaluated at the point
def plot_forward_difference_errors(f, x, exact, H):
    AbsError = []
    for h in H:
        approx = (f(x + h) - f(x)) / h
        AbsError.append(abs(approx - exact))

    # Make a loglog plot
    plt.loglog(H, AbsError, "r*")  # Makes axes log scales
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs. h")
    plt.grid()
    plt.show()
    return


f = lambda x: np.sin(x) * (1 - x)
x = 1
exact = -np.sin(x)
H = [2 ** (-n) for n in range(1, 11)]
plot_forward_difference_errors(f, x, exact, H)

H1 = [3 ** (-n) for n in range(1, 11)]
plot_forward_difference_errors(f, x, exact, H1)

H2 = [5 ** (-n) for n in range(1, 11)]
plot_forward_difference_errors(f, x, exact, H2)

H3 = [10 ** (-n) for n in range(1, 11)]
plot_forward_difference_errors(f, x, exact, H3)


# 5.13
def ForwardDiff(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = x[1] - x[0]
    df = []
    for j in np.arange(len(x) - 1):  # Makes a list of the integers from 0 to N-1
        df.append((f(x[j + 1]) - f(x[j])) / h)
    return df


f = lambda x: np.sin(x)
exact_df = lambda x: np.cos(x)
a = 0
b = 2 * np.pi
N = 100

df = ForwardDiff(f, a, b, N)
x = np.linspace(a, b, N + 1)
plt.plot(x, f(x), "b", x, exact_df(x), "r--", x[0:-1], df, "k-.")
# Plot approximation excluding the final value in our set of points as we cant compute this
plt.grid()
plt.legend(["f(x) = sin(x)", "exact first deriv", "approx first deriv"])
plt.show()


# 5.14
def ForwardDiff2(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = x[1] - x[0]
    y = f(x)
    df = (y[1:] - y[:-1]) / h
    return df


df = ForwardDiff2(f, a, b, N)
plt.plot(x, f(x), "b", x, exact_df(x), "r--", x[0:-1], df, "k-.")
# Plot approximation excluding the final value in our set of points as we cant compute this
plt.grid()
plt.legend(["f(x) = sin(x)", "exact first deriv", "approx first deriv"])
plt.show()

# 5.15
f1 = lambda x: np.sin(x) * (1 - x)
a = 0
b = 15
N = 250
x = np.linspace(a, b, N + 1)
y = f1(x)
df = ForwardDiff2(f1, a, b, N)
exact = lambda x: np.cos(x) * (1 - x) - np.sin(x)
fig, ax = plt.subplots(1, 2)  # Makes 1x2 grid for the plots to display in
ax[0].plot(x, y, "b", x[0:-1], df, "r--")  # Plots the two in the first box
ax[0].grid()
ax[1].semilogy(x[0:-1], abs(exact(x[0:-1]) - df))  # Second box, y axis log scale,
# plot error against x
ax[1].grid()
plt.show()

# 5.16
N = 150
df = ForwardDiff2(f1, a, b, N)
x = np.linspace(a, b, N + 1)
errors = abs(exact(x[0:-1]) - df)
maxError = max(errors)
print(maxError)


def plotMaxForwardDiffErrors(f, fPrime, a, b, H):
    maxError = []
    for i in H:
        N = int((b - a) / i)
        x = np.linspace(a, b, N + 1)
        df = ForwardDiff2(f, a, b, N)
        errors = abs(fPrime(x[0:-1]) - df)
        maxError.append(max(errors))
    plt.loglog(H, maxError, "r*-")  # Makes axes log scales
    plt.xlabel("h")
    plt.ylabel("Max Absolute Error")
    plt.title("Max Absolute Error vs. h")
    plt.grid()
    plt.show()
    return


plotMaxForwardDiffErrors(f1, exact, 0, 15, [1, 0.5, 0.2, 0.1])


# 5.18
def CentralDifference(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = 2 * (x[1] - x[0])
    y = f(x)
    df = (y[2:] - y[:-2]) / h
    return df


def plotCentralDifferenceErrors(f, x, exact, H):
    AbsError = []
    for h in H:
        approx = (f(x + h) - f(x - h)) / (2 * h)
        AbsError.append(abs(approx - exact))

    # Make a loglog plot
    plt.loglog(H, AbsError, "r*")  # Makes axes log scales
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs. h")
    plt.grid()
    plt.show()
    return


f = lambda x: np.sin(x) * (1 - x)
x = 1
exact = -np.sin(x)
delta_f = lambda h: (f(1 + h) - f(1 - h)) / (2 * h)
table(f, exact, delta_f)
H = [2 ** (-n) for n in range(1, 11)]
plotCentralDifferenceErrors(f, x, exact, H)

H1 = [3 ** (-n) for n in range(1, 11)]
plotCentralDifferenceErrors(f, x, exact, H1)

H2 = [5 ** (-n) for n in range(1, 11)]
plotCentralDifferenceErrors(f, x, exact, H2)

H3 = [10 ** (-n) for n in range(1, 11)]
plotCentralDifferenceErrors(f, x, exact, H3)

# 5.20
f = lambda x: np.sin(x) * (1 - x)
df = lambda x: np.cos(x) * (1 - x) - np.sin(x)
a = 0
b = 2 * np.pi

m = 16  # Number of different step sizes to plot
# Pre-allocate vectors for errors
fd_error = np.zeros(m)
cd_error = np.zeros(m)
# Pre-allocate vector for step sizes
H = np.zeros(m)

# Loop over the different step sizes
for n in range(m):
    N = 2 ** (n + 2)  # Number of subintervals
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = x[1] - x[0]  # step size

    # Calculate the derivative and approximations
    exact = df(x)
    forward_diff = (y[1:] - y[:-1]) / h
    central_diff = (y[2:] - y[:-2]) / (2 * h)

    # save the maximum of the errors for this step size
    fd_error[n] = max(abs(forward_diff - df(x[:-1])))
    cd_error[n] = max(abs(central_diff - df(x[1:-1])))
    H[n] = h

# Make a loglog plot of the errors agains step size
plt.loglog(H, fd_error, "b-", label="Forward Diff")
plt.loglog(H, cd_error, "r-", label="Central Diff")
plt.xlabel("Steps size h")
plt.ylabel("Maximum Absolute Error")
plt.title("Comparing Two First Derivative Approximations")
plt.grid()
plt.legend()
plt.show()


# 5.23
def SecondDiff(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = (x[1] - x[0]) ** 2
    y = f(x)
    df = (y[2:] - 2 * y[1:-1] + y[:-2]) / h
    return df


# 5.24
f = lambda x: np.sin(x) * (1 - x)
print(SecondDiff(f, 0, 2, 100)[49])
print(-2 * np.cos(1))

# 5.25
x = np.linspace(0, 15, 151)
exact = lambda x: -2 * np.cos(x) + (x - 1) * np.sin(x)
error = abs(SecondDiff(f, 0, 15, 150) - exact(x)[1:-1])
print(max(error))


def plotMaxSecondDiffErrors(f, fddash, a, b, H):
    maxError = []
    for i in H:
        N = int((b - a) / i)
        x = np.linspace(a, b, N + 1)
        df = SecondDiff(f, a, b, N)
        errors = abs(fddash(x[1:-1]) - df)
        maxError.append(max(errors))
    plt.loglog(H, maxError, "r*-")  # Makes axes log scales
    plt.xlabel("h")
    plt.ylabel("Max Absolute Error")
    plt.title("Max Absolute Error vs. h")
    plt.grid()
    plt.show()
    return


fddash = lambda x: (x - 1) * np.sin(x) - 2 * np.cos(x)
plotMaxSecondDiffErrors(f, fddash, 0, 15, [0.1, 0.2, 0.4, 0.8, 1.6])


f1 = lambda x: np.sin(x) * (1 - x)
a = 0
b = 15
N = 250
x = np.linspace(a, b, N)
y = f1(x)
df = SecondDiff(f1, a, b, N)
exact = lambda x: np.cos(x) * (1 - x) - np.sin(x)
fig, ax = plt.subplots(1, 2)  # Makes 1x2 grid for the plots to display in
ax[0].plot(x, y, "b", x[0:-1], df, "r--")  # Plots the two in the first box
ax[0].grid()
ax[1].semilogy(x[0:-1], abs(exact(x[0:-1]) - df))  # Second box, y axis log scale,
# plot error against x
ax[1].grid()
plt.show()
