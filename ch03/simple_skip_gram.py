import sys
sys.path.append("..")
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size) -> None:
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(hidden_size, vocab_size).astype('f')

        self.in_layer = MatMul(W_in)
        # skip_gramでは、出力層が複数存在するが、それぞれの出力層の入力と重みは同じになるため
        # コードでは、出力層は一つしか用意しない
        self.out_layer = MatMul(W_out)

        # 出力層が複数あることを表現するために損失のみ複数用意する
        # 正解ラベルのcontextのみが異なる
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        # 重みと勾配をリストにまとめる
        self.layers = [self.in_layer, self.out_layer, self.loss_layer0, self.loss_layer1]
        self.params, self.grads = [], []

        for layer in self.layers:
            # self.params.append(layer.params)
            # self.grads.append(layer.grads)

            self.params += layer.params
            self.grads += layer.grads

        self.word_vec = W_in


    def forward(self, target, corpus):
        h1 = self.in_layer.forward(target)
        h2 = self.out_layer.forward(h1)

        ls0 = self.loss_layer0.forward(h2, corpus[:, 0])
        ls1 = self.loss_layer1.forward(h2, corpus[:, 1])

        loss = ls0 + ls1

        return loss
    
    def backward(self):
        dout = 1
        ds0 = self.loss_layer0.backward(dout)
        ds1 = self.loss_layer1.backward(dout)

        ds = ds0 + ds1

        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)

        return None

