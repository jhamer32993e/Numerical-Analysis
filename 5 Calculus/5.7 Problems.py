import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad


# 5.48
def FirstDerivFromTable(data):
    array = np.array(data)
    x = array[:, 0]
    y = array[:, 1]
    dy = np.diff(y)  # difference between successive points in y
    dx = np.diff(x)
    yDash = dy / dx

    return list(zip(x[:-1], yDash))


# zip combines lists into 2D array ish
# list forces the zip output to be accessable as a list

x = np.linspace(0, 10, 101)
y = x**2
dataFrameOut = pd.DataFrame({"x": x, "y": y})
dataFrameOut.to_csv("test.csv", index=False)

dataFrameIn = pd.read_csv("test.csv")
dataPoints = list(zip(dataFrameIn["x"], dataFrameIn["y"]))

derivPoints = FirstDerivFromTable(dataPoints)
xDash, yDash = zip(*derivPoints)  # zip(*) unzips the array into separate arrays

plt.figure(figsize=(10, 8))

# Top subplot: Original function
plt.subplot(2, 1, 1)
plt.plot(x, y, "bo-", label="Original Data ($y = x^2$)")
plt.title("Underlying Function")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# Bottom subplot: Derivative approximation vs. Analytical exact
plt.subplot(2, 1, 2)
# Plot the numerical approximation
plt.plot(xDash, yDash, "ro", label="Numerical Approx (Forward Difference)")
# Plot the exact analytical derivative for comparison
plt.plot(x, 2 * x, "k--", label="Exact Analytical ($y' = 2x$)", alpha=0.6)
plt.title("First Derivative Approximation")
plt.xlabel("x")
plt.ylabel("y'")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# 5.49
def SecondDerivFromTable(data):
    array = np.array(data)
    x = array[:, 0]
    y = array[:, 1]
    h = x[1] - x[0]
    yDDash = (y[2:] - 2 * y[1:-1] + y[:-2]) / (h**2)

    return list(zip(x[1:-1], yDDash))


SecDerivPoints = SecondDerivFromTable(dataPoints)
xDDash, yDDash = zip(*SecDerivPoints)

plt.figure(figsize=(10, 8))

# Top subplot: Original function
plt.subplot(2, 1, 1)
plt.plot(x, y, "bo-", label="Original Data ($y = x^2$)")
plt.title("Underlying Function")
plt.ylabel("y")
plt.grid(True)
plt.legend()

# Bottom subplot: Derivative approximation vs. Analytical exact
plt.subplot(2, 1, 2)
# Plot the numerical approximation
plt.plot(xDDash, yDDash, "ro", label="Numerical Approx (Forward Difference)")
# Plot the exact analytical derivative for comparison
plt.plot(x, np.full_like(x, 2), "k--", label="Exact Analytical ($y'' = 2$)", alpha=0.6)
plt.title("Second Derivative Approximation")
plt.xlabel("x")
plt.ylabel("y''")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.ylim(0, 4)
plt.show()


# 5.50
def TrapeziumFromTable(data):
    array = np.array(data)
    x = array[:, 0]
    y = array[:, 1]
    dx = np.diff(x)
    Areas = 0.5 * dx * (y[:-1] + y[1:])
    return Areas


x2 = np.linspace(0, 10, 4)
y2 = x2**2
dataFrameOut2 = pd.DataFrame({"x": x2, "y": y2})
dataFrameOut2.to_csv("test2.csv", index=False)

dataFrameIn2 = pd.read_csv("test2.csv")
dataPoints2 = list(zip(dataFrameIn2["x"], dataFrameIn2["y"]))

array = np.array(dataPoints2)
x2 = array[:, 0]
y2 = array[:, 1]

Areas = TrapeziumFromTable(dataPoints2)
ApproxInt = np.concatenate(([0], np.cumsum(Areas)))
xSmooth = np.linspace(0, 10, 101)
ySmooth = xSmooth**2
exactIntegral = xSmooth**3 / 3

plt.figure(figsize=(10, 8))

# Top subplot: Original function and visually filled trapezoids
plt.subplot(2, 1, 1)
plt.plot(xSmooth, ySmooth, "k-", label="Exact Function ($y = x^2$)")

# Loop to draw each trapezoid individually so we can see what the function calculated
for i in range(len(x2) - 1):
    plt.fill_between(
        [x2[i], x2[i + 1]],
        [0, 0],
        [y2[i], y2[i + 1]],
        color="skyblue",
        alpha=0.5,
        edgecolor="blue",
    )

plt.plot(x2, y2, "bo", label="Data Points")
plt.title("Underlying Function with Trapezoidal Areas")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(
    xSmooth,
    exactIntegral,
    "k--",
    label="Exact Analytical Integral ($y = x^3/3$)",
    alpha=0.6,
)
plt.plot(x2, ApproxInt, "ro", label="Numerical Approx (Your Function)")
plt.title("Cumulative Integral Approximation")
plt.xlabel("x")
plt.ylabel("Cumulative Area")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 5.51
data = np.array(
    pd.read_csv(
        "https://github.com/gustavdelius/NumericalAnalysis2025/raw/main/data/Calculus/waterflow.csv"
    )
)
print(sum(TrapeziumFromTable(data)))
print("-" * 80)


def Trapezium(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = (b - a) / N
    y = f(x)
    Area = (h * 0.5) * (y[0] + y[-1] + 2 * np.sum(y[1:-1]))
    return Area


f = lambda x: 22.8 / (3.5 + 7 * ((x - 1.25) ** 4))
FExact = lambda x: (
    ((57 * 2**0.25) / 70)
    * np.log(
        np.abs(
            ((x - 1.25) ** 2 + 2**0.25 * (x - 1.25) + np.sqrt(2) / 2)
            / ((x - 1.25) ** 2 - 2**0.25 * (x - 1.25) + np.sqrt(2) / 2)
        )
    )
    + ((57 * 2**0.25) / 35)
    * (np.arctan(2**0.75 * (x - 1.25) + 1) + np.arctan(2**0.75 * (x - 1.25) - 1))
)

print("Approx:", Trapezium(f, 0, 2, 100))
Exact = FExact(2) - FExact(0)
print("Exact:", Exact)
print("Error:", abs((Trapezium(f, 0, 2, 100) - Exact) / Exact) * 100, "%")


# 5.52
def PlotTrapeziumErrors(f, a, b, exact, DX):
    Error = []

    for dx in DX:
        N = int(np.round((b - a) / dx))
        Error.append(abs(Trapezium(f, a, b, N) - exact))

    plt.loglog(DX, Error, "r^-", label="Trapezium Rule", markersize=8)

    plt.xlabel("Δx")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs. Δx")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    return


f1 = lambda x: np.exp(-(x**2) / 2)
f1Exact = quad(f1, -2, 2)[0]
DX = [2 ** (-n) for n in range(1, 11)]
PlotTrapeziumErrors(f1, -2, 2, f1Exact, DX)

f2 = lambda x: np.cos(x**2)
f2Exact = quad(f2, 0, 1)[0]
PlotTrapeziumErrors(f2, 0, 1, f2Exact, DX)


# 5.53
FuelPriceFrame = pd.read_csv("Gasoline.csv")
Week = [i for i in range(1, 478)]
FuelPriceAvg = FuelPriceFrame["New York State Average ($/gal)"]
FuelROC = FuelPriceAvg.values[1:] - FuelPriceAvg.values[:-1]

fig, ax1 = plt.subplots()
ax1.set_xlabel("Week Number")
ax1.set_ylabel("Price ($ / gallon)", color="blue")
ax1.plot(Week, FuelPriceAvg, color="blue", label="Gas Price", linewidth=2)
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, alpha=0.3)
plt.title("New York Gas Prices")

fig, ax2 = plt.subplots()
ax2.set_ylabel("Derivative ($ change per week)", color="red")
ax2.plot(Week[:-1], FuelROC, color="red", label="Rate of Change", alpha=0.6)
ax2.tick_params(axis="y", labelcolor="red")
ax2.axhline(0, color="black", linestyle="--", linewidth=1)

plt.title("New York Gas Prices Rate of Change")
fig.tight_layout()
plt.show()
# Derivative shows the rate of change of gas prices in New York state by week


Taxi = pd.read_csv("Taxi.csv", nrows=10000)
Distance = Taxi["trip_distance"].fillna(0).values
JourneyNo = [i for i in range(len(Distance))]


def TrapeziumArray(y, x):
    y = np.array(y)
    x = np.array(x)

    dx = np.diff(x)
    Areas = 0.5 * dx * (y[:-1] + y[1:])
    return Areas


print(sum(TrapeziumArray(Distance, JourneyNo)))
# Integral gives the total distance travelled by taxis in 10000 journeys

# 5.54
f1 = lambda x: x / (x**4 + 1)
F1Exact = 0.5 * np.arctan(4) - 0.5 * np.arctan(1)
f2 = lambda x: (x - 1) ** 3 * (x - 2) ** 2
F2Exact = -549 / 20
f3 = lambda x: np.sin(x**2)
F3Exact = quad(f3, -1, 2)[0]


def ForwardDiff(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = x[1] - x[0]
    df = []
    # Fixed the loop structure to be more pythonic
    for j in range(len(x) - 1):
        df.append((f(x[j + 1]) - f(x[j])) / h)
    return df


def Simpson(f, a, b, N):
    x = np.linspace(a, b, N + 1)
    h = (b - a) / N
    y = f(x)
    m = (x[1:] + x[:-1]) / 2
    Area = (h / 6) * (y[0] + y[-1] + 2 * np.sum(y[1:-1]) + 4 * np.sum(f(m)))
    return Area


def IntAndDeriv(f, FExact, a, b, N, title_label="Function"):
    Area = Simpson(f, a, b, N)
    print(f"--- {title_label} ---")
    print("Approx Area:", Area)
    print("Error:", abs(Area - FExact))

    DerivPoints = ForwardDiff(f, a, b, N)
    x = np.linspace(a, b, N + 1)

    plt.figure()

    plt.plot(x, f(x), "b", label="f(x)")
    plt.plot(x[:-1], DerivPoints, "k-.", label="Approx first deriv")
    plt.grid()
    plt.legend()
    plt.title(title_label)
    plt.show(block=False)


IntAndDeriv(f1, F1Exact, -1, 2, 1000, "f1")
IntAndDeriv(f2, F2Exact, -1, 2, 1000, "f2")
IntAndDeriv(f3, F3Exact, -1, 2, 1000, "f3")


# 5.55
data = np.array(
    pd.read_csv(
        "https://github.com/gustavdelius/NumericalAnalysis2025/raw/main/data/Calculus/bikespeed.csv"
    )
)
t = data[:, 0]
v = data[:, 1]
print(sum(TrapeziumArray(v, t)))
