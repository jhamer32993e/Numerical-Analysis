import numpy as np
import matplotlib.pyplot as plt


# 10.8
def coeffMatrix(numX, a, LeftBCType, RightBCType):
    N = numX + 1
    A = np.zeros((N, N))

    for i in range(1, N - 1):
        A[i, i - 1] = -a
        A[i, i] = 1 + 2 * a
        A[i, i + 1] = -a

    if LeftBCType == "D":
        A[0, 0] = 1
    elif LeftBCType == "N":
        A[0, 0] = 1
        A[0, 1] = -1

    if RightBCType == "D":
        A[-1, -1] = 1
    elif RightBCType == "N":
        A[-1, -2] = -1
        A[-1, -1] = 1

    return A


def BackwardsHeat(
    k, t0, tmax, numT, x0, xmax, numX, IC, LeftBCType, LeftBC, RightBCType, RightBC
):
    x = np.linspace(x0, xmax, numX + 1)
    dx = x[1] - x[0]
    t = np.linspace(t0, tmax, numT + 1)
    dt = t[1] - t[0]

    a = k * dt / dx**2
    U = np.zeros((len(x), len(t)))

    U[:, 0] = IC(x)
    A = coeffMatrix(numX, a, LeftBCType, RightBCType)

    for i in range(len(t) - 1):
        b = np.copy(U[:, i])

        if LeftBCType == "D":
            U[0, i + 1] = LeftBC(t[i + 1])

        elif LeftBCType == "N":
            q = LeftBC(t[i + 1])
            U[0, i + 1] = U[1, i + 1] - (q * dx)

        if RightBCType == "D":
            U[-1, i + 1] = RightBC(t[i + 1])
        elif RightBCType == "N":
            q = RightBC(t[i + 1])
            U[-1, i + 1] = U[-2, i + 1] + (q * dx)

        U[:, i + 1] = np.linalg.solve(A, b)
    return x, t, U


# 10.9
x, t, U = BackwardsHeat(
    0.2,
    0,
    1,
    2,
    0,
    1,
    20,
    lambda x: np.sin(np.pi * x),
    "D",
    lambda t: 0,
    "D",
    lambda t: 0,
)

print(U[12, -1])

print("- " * 40)

# 10.10
x, t, U = BackwardsHeat(
    0.5, 0, 4, 10, 0, 2, 20, lambda x: x**2, "D", lambda t: t, "D", lambda t: 4 - t
)

print(U[10, -1])

print("- " * 40)


# 10.11
def coeffMatrixA(numX, a, LeftBCType, RightBCType):
    N = numX + 1
    A = np.zeros((N, N))

    r = a / 2

    for i in range(1, N - 1):
        A[i, i - 1] = -r
        A[i, i] = 1 + 2 * r
        A[i, i + 1] = -r

    if LeftBCType == "D":
        A[0, 0] = 1
    elif LeftBCType == "N":
        A[0, 0] = 1
        A[0, 1] = -1

    if RightBCType == "D":
        A[-1, -1] = 1
    elif RightBCType == "N":
        A[-1, -2] = -1
        A[-1, -1] = 1

    return A


def coeffMatrixB(numX, a):
    N = numX + 1
    B = np.zeros((N, N))
    r = a / 2

    for i in range(1, N - 1):
        B[i, i - 1] = r
        B[i, i] = 1 - 2 * r
        B[i, i + 1] = r

    return B


def CrankNicolson(
    k, t0, tmax, numT, x0, xmax, numX, IC, LeftBCType, LeftBC, RightBCType, RightBC
):
    x = np.linspace(x0, xmax, numX + 1)
    dx = x[1] - x[0]
    t = np.linspace(t0, tmax, numT + 1)
    dt = t[1] - t[0]

    a = k * dt / dx**2
    A = coeffMatrixA(numX, a, LeftBCType, RightBCType)
    B = coeffMatrixB(numX, a)

    U = np.zeros((len(x), len(t)))
    U[:, 0] = IC(x)

    for i in range(len(t) - 1):
        b = np.dot(B, U[:, i])

        if LeftBCType == "D":
            b[0] = LeftBC(t[i + 1])
        elif LeftBCType == "N":
            b[0] = -LeftBC(t[i + 1]) * dx

        if RightBCType == "D":
            b[-1] = RightBC(t[i + 1])
        elif RightBCType == "N":
            b[-1] = RightBC(t[i + 1]) * dx

        U[:, i + 1] = np.linalg.solve(A, b)

    return x, t, U


# 10.12
x, t, U = CrankNicolson(
    0.2,
    0,
    1,
    2,
    0,
    1,
    20,
    lambda x: np.sin(np.pi * x),
    "D",
    lambda t: 0,
    "D",
    lambda t: 0,
)

print(U[12, -1])

print("- " * 40)

# 10.13
x, t, U = CrankNicolson(
    0.5, 0, 4, 10, 0, 2, 40, lambda x: x**2, "D", lambda t: t, "D", lambda t: 4 - t
)
print(U[20, -1])
