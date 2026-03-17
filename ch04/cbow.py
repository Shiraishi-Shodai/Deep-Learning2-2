import sys
sys.path.append("..")
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingLoss
import numpy as np

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # 入力層
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        
        # 出力層
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 入力層と出力層の重みと購買を配列にまとめる
        layers = self.in_layers + [self.ns_loss]

        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 分散表現を設定
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # contextは二次元
        # targetは一次元
        h = 0

        for i, layer in enumerate(self.in_layers):
            # 各入力層で、入力値のコンテキストとして与えられた単語の重みを抜き出し、加算する
            h += layer.forward(contexts[:, i])
        
        # 入力値と入力層の重みを掛けた値を平均化
        h *= 1 / len(self.in_layers)
        # 出力層で、正例と負例両方のデータでスコアと損失関数をそれぞれ計算(バッチサイズ、正例 + 負例)
        loss = self.ns_loss.forward(h, target)

        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward()
        dout *= 1 / len(self.in_layers)

        for layer in self.in_layers:
            layer.backward(dout)

        return None
    