import numpy as np
import matplotlib.pyplot as plt


# 5.29
def RiemannSum(f, a, b, N, method="left"):
    x = np.linspace(a, b, N + 1)
    y = f(x)
    w = (b - a) / N
    area = 0
    if method == "left":
        area = sum(w * y[:-1])
    elif method == "right":
        area = sum(w * y[1:])
    elif method == "midpoint":
        area = sum(0.5 * w * (y[:-1] + y[1:]))
    return area


f = lambda x: x**2
f1 = lambda x: x**3
print(RiemannSum(f, 0, 1, 100, "left"))
print(RiemannSum(f, 0, 1, 100, "right"))
print(RiemannSum(f, 0, 1, 100, "midpoint"))
print(RiemannSum(f1, 0, 1, 100, "left"))
print(RiemannSum(f1, 0, 1, 100, "right"))
print(RiemannSum(f1, 0, 1, 100, "midpoint"))
print("-" * 50)

# 5.30
exact = np.cos(0) - np.cos(1)
print("Exact:", exact)
f2 = lambda x: np.sin(x)
print("Approximation:", RiemannSum(f2, 0, 1, 100, "left"))
print()


def table(f, a, b, factor, method="left"):
    hdr1 = "Δx"
    hdr2 = f"A_{method}(Δx)"
    hdr3 = f"|I - A_{method}(Δx)|"
    hdr4 = "Error reduction factor"
    print(f"{hdr1:<20} | {hdr2:<18} | {hdr3:<18} | {hdr4:<18}")
    print("-" * 82)
    for n in range(1, 11):
        dx = factor ** (-n)
        dx_str = f"3^-{n} = {dx:g}"
        approx = RiemannSum(f2, a, b, int(np.round(1 / dx)), method="left")
        error = abs(exact - approx)
        if n == 1:
            print(f"{dx_str:<20} | {approx:<18.15f} | {error:<18.15f} | ")
        else:
            print(
                f"{dx_str:<20} | {approx:<18.15f} | {error:<18.15f} | {error_prev/error:<18.15f}"
            )
        error_prev = error
    print()
    return


# table(f2, 0, 1, 3, "left")
# table(f2, 0, 1, 3, "right")
# table(f2, 0, 1, 3, "midpoint")


# 5.31
def PlotRiemannErrors(f, a, b, exact, DX):
    LeftError = []
    RightError = []
    MidpointError = []

    for dx in DX:
        N = int(np.round((b - a) / dx))
        LeftError.append(abs(RiemannSum(f, a, b, N, "left")))
        RightError.apend(abs(RiemannSum(f, a, b, N, "right")))
        MidpointError.append(abs(RiemannSum(f, a, b, N, "midpoint")))

    plt.loglog(DX, LeftError, "b*", label="Left Sum", markersize=14)
    plt.loglog(DX, RightError, "ro", label="Right Sum")
    plt.loglog(DX, MidpointError, "g+", label="Midpoint Sum")

    plt.xlabel("Δx")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs. Δx")
    plt.legend()
    plt.grid(True)
    plt.show()
    return
