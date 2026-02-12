import numpy as np

#1.24
accounts = 100 + (100000 - 100) * np.random.rand(50000, 1)
accounts = np.floor(100 * accounts) / 100

illegal = 0
days = 0
while illegal < 10**6:
    accounts = accounts * (100 + 5 / 365) / 100
    illegal += np.sum(accounts - np.floor(100 * accounts) / 100)
    accounts = np.floor(100 * accounts) / 100
    days += 1
print(illegal)
print(days)

#1.26
print(9^5)
print(5^9)
print(9^2)
print(9^3)
print(9^4)