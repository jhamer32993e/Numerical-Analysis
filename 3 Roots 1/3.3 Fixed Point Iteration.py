import numpy as np

f1 = lambda x: np.cos(x)
x1 = 0.2
for i in range(10):
    print(x1)
    x1 = np.cos(x1)
print(x1)
print("Estimated solution of cos(x)-x=0 is:", x1)
