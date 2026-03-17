import numpy as np
import copy
import collections
import sys
import itertools
import sys
sys.path.append("..")
from matplotlib import pyplot as plt
from matplotlib import image as mping
import japanize_matplotlib

# h = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) # (2, 3)
# w = np.arange(12).reshape(3, -1) # (3, 4)
# # print(w)

# idx = np.array([0, 1])

# print(w[:, idx])print(w)
# print(h @ w[:, idx])
# print(w[:, 0])
# print()
# print(h[0] @ w[:, 0])
# print(np.sum(h * w[:, idx], axis=1))

# c = (1, 10)
# d, e = c
# print(d, e)

# dw = np.zeros((6, 3))
# dh = np.ones((3, 3))
# idx = np.array([0, 1, 0])

# np.add.at(dw, idx, dh)
# print(dw)

# for i in np.arange(0.1, 1, 0.1):
#     print(f"{i}: {-np.log(i)}")

# a = [0, 0, 1, 0]
# counter = collections.Counter(a)
# print(counter[0])
# print(len(counter))

# print(counter[0, 1])

# 元の配列
# a = np.arange(5)
# print("a:", a)

# # ビューを作成（メモリは共有している）
# b = a[::2]
# print("b (view):", b)

# c = np.copy(b)

# d = copy.deepcopy(b)
# # 元配列を書き換え
# a[0] = 99
# print("\nAfter modifying a[0] = 99")
# print("a:", a)
# print("b (view):", b)
# print("c (np.copy):", c)
# print("d (deepcopy):", d)


# ランダムチョイス
# a = np.random.choice(3, size=3, replace=False, p=[0.2, 0.5, 0.3])
# print(a)


# a = np.arange(10).reshape(2, -1).tolist()
# z = np.arange(10, 20).reshape(2, -1).tolist()
# w = np.arange(20, 30).reshape(2, -1).tolist()
# b = [a, z, w]

# # print(a)
# # print(b)

# c, v, x = b
# d = b

# print(c, type(c))
# print(v, type(v))
# print(x, type(x))
# print(d, type(d))


# grads = [np.zeros_like((2, 1)), np.zeros_like((2, 1)), np.zeros_like((2, 1))]
# print(*grads)


# a = np.arange(12).reshape(3, -1)
# print(a)
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=1))

# def test(a, b, c):
#     print(a)
#     print(b)
#     print(c)

# test_arr = [1, 2, 3]
# test(*test_arr)

# b = np.ones((2, 3, 2))
# print(b)
# print()

# print(b[:, 0, :].shape)

# print(np.empty((1, 2), dtype="int"))

# a = [1, 2, 3]
# b = [i + 1 for i in a]
# c = [i + 2 for i in a]

# print(a)
# print(b)
# print(c)

# grads = [0, 0, 0]

# for i in np.array([a, b, c]):
#     print(i)
#     grads[0] += i

# print(grads)


# e = [1, 2, 3] + [1, 2, 3]
# print(e)

# print(np.random.randn(1))

# print(1 % 99)

# a = np.arange(12)
# b = [1, 2, 3]
# print(a[b])

# c = a.tolist()
# print(c, type(c))
# print(c[b])

# a = [[1, 2], [5, 6]]
# b = [[3, 4], [10, 11]]
# print(a + b)

# print(None * np.arange(10))

# x = np.linspace(1, 1000, 1000)
# t = np.linspace(1000, 2000, 1000)
# data_size = x.shape[0]
# time_size = 35
# time_idx = 0
# batch_size = 20
# jump = data_size // batch_size

# offsets = [i * jump for i in range(batch_size)]

# batch_x = np.empty((batch_size, time_size), dtype='i')
# batch_t = np.empty((batch_size, time_size), dtype='i')

# for time in range(time_size):
#     for i, offset in enumerate(offsets):
#         batch_x[i, time] = x[(offset + time_idx) % data_size]
#         batch_t[i, time] = t[(offset + time_idx) % data_size]
#     time_idx += 1

# print(x)
# print(offsets)
# print(batch_x)

# def seki(grads):
#     for grad in grads:
#         grad *= 0.1

# a = [np.arange(10, dtype="f"), np.arange(10, 20, dtype="f")]
# print(a)
# seki(a)
# print(a)

# a, b = [1, 2]
# print(a, b)

# a = np.arange(12).reshape(3, -1)
# dropout_ratio = 0.5

# mask = np.random.rand(*a.shape) > dropout_ratio
# print(mask)

# p = np.array([0.3, 0.5, 0.2])
# for i in range(30):
#     print(np.random.choice(len(p), size=1, p=p))

# a = np.arange(12).reshape(2, 2, 3)
# print(a)

# b = a.reshape(2 * 2, -1)
# print(b)

# c = b.reshape(2, 2, -1)
# print(c)

# text_np = np.array(["Hello", "Yes", "No", "Wow", "Hello"])
# corpus = []
# word_to_id = {}
# id_to_word = {}

# for text in text_np:
#     if text not in word_to_id:
#         new_id = len(id_to_word)
#         word_to_id[text] = new_id
#         id_to_word[new_id] = text
#     corpus.append(word_to_id[text])

# print(corpus)
# print(word_to_id)
# print(id_to_word)

# a = np.arange(1, 5).reshape(2, 2)
# b = a.reshape(2 * 2, -1)

# w = np.arange(10, 50, 10).reshape(1, -1)
# u = np.dot(b, w)
# print(u)
# print()
# print(u.reshape(2, 2, 4))

# def get_test_batch(x, t, batch_size, time_size, current_idx):
#     """
#     わりきれなかったデータは捨てる。
#     """
#     data_size = x.shape[0]
#     batch_x = np.zeros((batch_size, time_size))
#     batch_t = np.zeros((batch_size, time_size))

#     for b in range(batch_size):
#         for i in range(time_size):
#             batch_x[b, i] = x[current_idx + i]
#             batch_t[b, i] = t[current_idx + i]
#         current_idx += time_size

#     return batch_x, batch_t, current_idx

# # データの用意 (100)
# train_X = np.arange(1, 101).reshape(-1, 1)
# train_T = np.arange(200, 301)
# data_size = train_X.shape[0]
# batch_size = 5
# time_size = 3
# current_idx = 0

# max_iters = data_size // (batch_size * time_size)

# for iters in range(max_iters):
#     batch_x, batch_t, current_idx = get_test_batch(train_X, train_T, batch_size, time_size, current_idx)

# print(batch_x)
# print(batch_t)

# a = np.zeros((4, 5))
# b = np.arange(10).reshape(2, -1)

# c = np.arange(5).reshape(-1, 1)
# d = c[3, 0]
# print(d)

# print((np.random.randn(10000).flatten().sum()) / 10000)
# print((np.random.rand(10000).flatten().sum()) / 10000)

# print(np.array("x").reshape(1, 1).shape)
# a = np.arange(12).reshape(2, 2, 3)
# print(a)

# print()
# print(a.flatten())

# flg = np.array([True, True, False])
# print(flg)
# print(flg.astype(np.float32))

# a = np.array([10, 20, 30])
# print(np.argmax(a))

# a = np.arange(12).reshape(3, -1)
# print(a)


# print(a[:, :-1])

# print(a[:, 1:])

# print("Hello Zellij")

# a = np.arange(12).reshape(4, -1)
# print(a)
# print(a[[1]])
# print(a[1])

# b = 0.285
# print(f"{b*100:.3f}")

# a = [10, 5, 12]
# plt.plot(np.arange(len(a)), a, marker='o')
# plt.show()

# def show_result(axes_size, labels):

#     for row in range(axes_size):
#         for col in range(axes_size):
#             print(labels[row*axes_size + col])

# a = np.arange(16)
# show_result(4, a)

image = mping.imread("first-test.png")

# plt.style.use("ggplot")
# plt.style.use("seaborn-v0_8")
# plt.style.use("dark_background")

fig, ax = plt.subplots(2, 2)
fig.set_facecolor("#f0f0f0")
ax[0, 0].set_facecolor("#2d2d2d")    # グラフ部分
ax[1, 0].set_facecolor("#2d2d2d")    # グラフ部分
ax[0, 0].imshow(image)
ax[0, 0].text(
    0.5,
    -0.15,
    "これは1枚目",
    transform=ax[0, 0].transAxes,
    ha="center"
)
ax[0, 1].imshow(image)
ax[0, 1].text(
    0.5,
    -0.15,
    "これは1枚目",
    transform=ax[0, 1].transAxes,
    ha="center"
)

plt.tight_layout()

plt.show()