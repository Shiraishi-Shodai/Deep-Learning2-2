import numpy as np

# N, T, H = 10, 5, 4

# hs = np.random.randn(N, T, H)
# a = np.random.randn(N, T)

# ar = a.reshape(N, T, 1).repeat(H, axis=2)

# print(ar)
# t = hs * ar
# print(t.shape)

# c = np.sum(t, axis=1)
# print(c.shape)

N, T, H = 3, 2, 3

# a = np.random.randn(N, T)
# print(a)
# ar = a.reshape(N, T, 1).repeat(H, axis=2)
# print(ar)

# a = np.random.randn(N, 1, H).astype("i8")
# print(a)
# ar = a.repeat(T, axis=1)
# print(ar)