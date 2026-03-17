import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
from common.trainer import Trainer
from common.optimizer import Adam
from common.layers import MatMul, SoftmaxWithLoss
from common.util import preprocess, create_contexts_target, convert_one_hot
import numpy as np

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size) -> None:
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)

        self.loss_layer = SoftmaxWithLoss()
    
        layers = [self.in_layer0, self.in_layer1, self.out_layer]

        self.params, self.grads = [], []

        for layer in layers:
            # リストの足し算はリストの結合(numpyと違い要素同士の足し算ではない)
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = W_in
    
    def forward(self, contexts, target):
        # contexts (3, 2, 7)
        # contexts[:, 0] (3, 7)
        h0 = self.in_layer0.forward(contexts[:, 0]) 
        h1 = self.in_layer1.forward(contexts[:, 1])

        h = (h0 + h1) * 0.5

        h_out = self.out_layer.forward(h)
        loss = self.loss_layer.forward(h_out, target)

        return loss
    
    def backward(self):
        ds = self.loss_layer.backward()
        da = self.out_layer.backward(ds)

        da *= 0.5

        self.in_layer0.backward(da)
        self.in_layer1.backward(da)

        return None