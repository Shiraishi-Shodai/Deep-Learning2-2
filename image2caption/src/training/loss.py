import sys
from pathlib import Path

# プロジェクトルート (image2caption/) を取得
root = Path(__file__).resolve().parent.parent.parent
# プロジェクトルートとその親 (Deep-Learning2/) をパスに追加
sys.path.extend([str(root), str(root.parent)])

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from common.time_layers import SoftmaxWithLoss

criterion = nn.CrossEntropyLoss()

logits = torch.tensor([
                       [0.0, 100.0, 0.0]])

targets = torch.tensor([1])

loss = criterion(logits, targets)
# print(loss)

# s = np.exp(0) + np.exp(1) + np.exp(0)
# s1 = np.exp(1) / s
# print(s1)
# print(-np.log(s1))

# def softmax(x):
#     c = np.max(x)
#     exp_x = np.exp(x - c)
#     sum_exp_x = np.sum(exp_x)
#     d = exp_x / sum_exp_x
#     return d


def cross_entropy(x, t):
    d = np.log(x) * t
    return -np.sum(d)

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts]) # 
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする(True, Falseの二次元配列(N*T, 1)をかける)

        dx = dx.reshape((N, T, V))

        return dx


ts = torch.tensor([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
])

xs = torch.tensor([
        [
            [100, 0, 0, 0],
            [0, 100, 0, 0],
        ],
        [
            [0, 0, 100, 0],
            [0, 0, 0.9, 0.1],
        ]
])

if ts.ndim == 3:
    ts = ts.argmax(axis=2)

ignore_label = -1
mask = (ts != ignore_label)

N, T, V = xs.shape
xs = xs.reshape(N*T, V)
ts = ts.reshape(N*T)
mask = mask.reshape(N*T)

def softmax(x):
    if x.ndim == 2:
        x = x - xs.max(dim=1, keepdims=True).values
        x = torch.exp(x)
        x /= x.sum(dim=1, keepdims=True)
    elif x.ndim == 1:
        x = x - torch.max(x)
        x = torch.exp(x) / torch.sum(torch.exp(x))
    return x

ys = softmax(xs)
ls = torch.log(ys[torch.arange(N * T), ts])
# print(ys) 
# print(ls)

ls *= mask
loss = -torch.sum(ls)
loss /= mask.sum()
print(loss)
print(mask.sum())