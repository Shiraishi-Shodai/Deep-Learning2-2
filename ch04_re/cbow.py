import sys

from common import layers
sys.path.append("..")
from ch04_re.negtive_sampling_layer import NegativeSamplingLoss
from common.layers import Embedding
import numpy as np

class CBOW:
    def __init__(self, corpus, V, H, window_size, sample_size, power) -> None:
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        self.window_size = window_size

        self.in_layers = []

        # 入力層を定義
        for _ in range(2 * window_size):
            self.in_layers.append(Embedding(W_in))

        # 出力層を定義
        self.out_layer = NegativeSamplingLoss(W_out, corpus, sample_size=sample_size, power=power)

        self.params, self.grads = [], []
        layers = self.in_layers + [self.out_layer]

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 分散表現を表す変数を定義
        self.word_vec = W_in

    
    def forward(self, contexts, target):

        h = 0

        for i in range(2 * self.window_size):
            h += self.in_layers[i].forward(contexts[:, i])
        
        # 中間層の値をコンテキストサイズで平均する
        h *= (1 / self.window_size)

        loss = self.out_layer.forward(h, target)

        return loss

    
    def backward(self, dout=1):
        dh = self.out_layer.backward(dout)

        # forward時に1/widows_sizeを掛けたから、backwardでは、εに1/window_sizeを掛ける
        dh *= (1 / self.window_size)

        for layer in self.in_layers:
            layer.backward(dh)

        return None


