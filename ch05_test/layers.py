import sys
sys.path.append("..")
import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = W
        self.grads = np.zeros_like(W)
        self.idx = None
    
    def forward(self, idx):
        out = W[idx]
        return out
    
    def backward(self, dout, idx):
        W = self.params
        dW = self.grads
        dW[...] = 0

        np.add.at(dW, idx, dout)

        return None

class RNN:
    """バッチサイズごとにforward, backward
    出力： N * H
    """
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.cache = None

    def forward(self, x, h_pred):
        Wx, Wh, b = self.params

        t = np.dot(h_pred, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)
        self.cache = (x, h_pred, h_next)

        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_pred, h_next = self.cache
        
        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0) # 列ごと
        dWh = np.dot(h_pred.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, dWx)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = dWb

        return dx, dh_prev

class TimeRNN:
    def __init__(self, rnn_Wx, rnn_Wh, rnn_Wb, stateful=False):
        self.params = [rnn_Wx, rnn_Wh, rnn_Wb]
        self.grads = [np.zeros_like(rnn_Wx), np.zeros_like(rnn_Wh), np.zeros_like(rnn_Wb)]
        self.h = None
        self.dh = None
        self.layers = None
        self.stateful = stateful
    
    def forward(self, xs):
        rnn_Wx, rnn_Wh, rnn_Wb = self.params
        N, T, D = xs.shape
        D, H = rnn_Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs
    
    def backward(self, dhs):
        rnn_Wx, rnn_Wh, rnn_Wb = self.params
        N, T, H = dhs.shape
        D, H = rnn_Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0]

        for t reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            # 各RNNレイヤーごとに勾配を合算
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh
        return dxs
    
    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None





        

        

