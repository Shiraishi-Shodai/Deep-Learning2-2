import numpy as np

N, T, D = 2, 3, 4
a = np.random.randn(N, T, D)
a = a * 10
a = a.astype("i8")

print(a, end="\n")
print(a[:, -1])