import sys
from typing import Collection
sys.path.append("..")
import numpy as np
from common.layers import Embedding
from negtive_sampling_layer import EmbeddingDot
from collections import Counter

# a  = np.arange(12).reshape(3, 4)
# index = np.array([0, 2, 0])
# vocab_size = 2

# embed = Embedding(a)

# print(a)
# print()
# target_W = embed.forward(index)
# print(target_W)
# print()

# dout = np.arange(10, 130, 10).reshape(3, -1)
# print(dout)
# print()

# embed.backward(dout)
# print(embed.params)
# print(embed.grads)

"""
EmbeddingDotの順伝搬の確認
"""
# 入力の準備
# vocab_size = 5
# W_in = np.arange(15, dtype="f4").reshape(vocab_size, 3)
# context_index = np.array([0, 2, 0])

# # 入力層 -> 中間層の順伝搬
# embed = Embedding(W_in)
# h = embed.forward(context_index)
# print(h)

# W_out = np.arange(15, 30, dtype="f4").reshape(vocab_size, 3)

# print(f"重み: {W_out}")
# print()

# embedDot = EmbeddingDot(W_out)
# out = embedDot.forward(h, context_index)
# print("h2")
# print(out)


# """
# EmbeddingDotの逆伝搬の確認
# """

# dout = np.array([0.1, 0.2, 0.1])
# dh = embedDot.backward(dout)
# print(dh)
# print()
# print(embedDot.embed.grads)

# a = np.array([1, 0, 1, 0, 2, 9, 4, 10, 4, 4, 7, 38, 2, 0, 1], dtype=np.float32)

# counter = Counter(a)
# vocab_size = len(counter)
# p = list(counter.values())

# p /= np.sum(p)

# print(a.shape, p.shape)
# print(np.random.choice(vocab_size, size=5, replace=False, p=p))

# a = np.array([[4, 3], [5, 6]])
# print(np.linalg.norm(a, ord=2, axis=1, keepdims=True))

# idxs = np.array([0, 3, 2, 4, 50, 8])
# banned_ids = np.array([3, 2])
# k = 5

# # 除外処理
# keep = [i for i in range(len(idxs)) if idxs[i] not in banned_ids]
# idxs = [idxs[i] for i in keep]

# print(keep)
# print(idxs)

a = np.array([[0.14, 0.27],[0.56, 0.03]]).astype("f")
b = a * 0.6
c = a * 0.4

print(b)
print(c)
print(b.sum(), c.sum())