import numpy as np
import os
from matplotlib import pyplot as plt

index_size = 6
a = np.arange(20).reshape(10, -1).astype("f")
b = np.arange(0, 120, 10).reshape(index_size, -1)

rng = np.random.default_rng()

# print(a, end="\n")
# print(b)

# data_size = a.shape[0]
# index = np.arange(index_size)

# print(f"サイズ: {data_size}")
# print(f"インデックス: {index}")

# # 書き込まれる側, 書き込み先, 書き込むデータ
# np.add.at(a, index, b)
# print(a)

# print(rng.integers(0, 3, size=(3, 3), endpoint=True))
# print(os.name)
# colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}

# print(colors["ok"] + colors["fail"])

# x = 10
# dropout_ratio = np.arange(0, 1.1, 0.1)
# y = x * (1.0 - dropout_ratio)

# print(dropout_ratio)
# print(y)
# plt.plot(np.arange(len(y)), y, marker="o", color="orange")
# plt.show()

N, T, H = 2, 3, 4

# a = np.arange(24).reshape(N, T, H)
# h = a[:, -1, :]
# hs = np.repeat(h, T, axis=0)
# print(a, end="\n")
# print(h, end="\n")
# print("repeat実行")
# print(hs, end="\n")
# print("reshape実行")
# print(hs.reshape(N, T, H), end="\n")

# b = np.arange(3).reshape(1, -1)
# print(np.repeat(b, T, axis=1).reshape(T, -1))

# a = np.arange(N*T*H).reshape(N, T, H)
# b = a + 10

# print(a)
# print(np.concatenate((a, b), axis=2))

# dy = np.array([[1, 0.2]])
# W = np.arange(0.1, 0.7, 0.1).reshape(3, -1)

# print(dy @ W.T)

# x = np.arange(1, 5).reshape(2, -1)
# W = np.arange(0.1, 0.7, 0.1).reshape(2, -1)
# # y = x @ W
# # print(y)

# dy = np.flipud(np.arange(1, 7)).reshape(2, -1)

# dx = dy @ W.T
# print(dx)

a = np.arange(6).reshape(2, 3)
print(a)
print(a.reshape(2*3))