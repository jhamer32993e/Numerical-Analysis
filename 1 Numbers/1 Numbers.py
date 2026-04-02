# 1.3
x = 0.1
print(f"x={x}")
for i in range(0, 15):
    if x > 0 and x < 0.5:
        x = 2 * x
    elif x >= 0.5 and x < 1:
        x = 2 * x - 1
    else:
        break
    print(x)

print("- " * 40)

# 1.4
y = 0.125
print(f"y={y}")
for i in range(0, 50):
    if y > 0 and y < 0.5:
        y = 2 * y
    elif y >= 0.5 and y < 1:
        y = 2 * y - 1
    else:
        break
    print(y)
