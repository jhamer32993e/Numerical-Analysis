import numpy as np
import matplotlib.pyplot as plt


f = lambda x: np.sin(x) * (1 - x)
exact = -np.sin(1)  # Derivative evaluated at 1
delta_f = lambda h: (f(1 + h) - f(1)) / h

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
