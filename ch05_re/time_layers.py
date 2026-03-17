import numpy as np

from common.layers import Embedding
from common.functions import softmax

class RNN:
    def __init__(self, Wx, Wh, b, lam=1):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.lam = lam
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        # dWx = np.dot(x.T, dt)
        dWx = np.dot(x.T, dt) + self.lam * Wx
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful
    
    def set_state(self, h):
        self.h = h
    
    def reset_state(self):
        self.h = None
    
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        
        for t in range(T):
            layer = RNN(*self.params)
            # h時点の順伝搬(バッチサイズごとにRNNレイヤが順伝搬を行う)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape # 逆伝搬したデータの形
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0] # TimeRNNの勾配(1時データ用)
        
        # 1データずつ逆伝搬し、dxsを更新
        for t in reversed(range(T)):
            layer = self.layers[t]
            # h時点の逆伝搬(バッチサイズごとに)
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            # 1データ分(h時点)のWx, Wh, bを最終的な勾配に加算
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        # TimeRNNの勾配を
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad # 中身のデータだけ変更する
        self.dh = dh

        return dxs


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            # h時点(batchサイズごとに単語の埋め込みベクトルを計算)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):

        """元の計算方法
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)
        """

        # 形状の取得方法を入力値の次元によって変更(TimeAffineをレイヤーの最初にする場合とそれ以外の場合で入力値の次元はことなるから)
        if x.ndim == 3:
            N, T, D = x.shape
            W, b = self.params
        
        elif x.ndim == 2:
            N, T = x.shape
            W, b = self.params
            _, D = W.shape

        # 行方向に行列を伸ばす
        rx = x.reshape(N * T, -1)
        # rx(N * T, -1) ⦿ (1, D) + (D)
        out = np.dot(rx, W) + b

        self.x = x

        # rx(N, T, D)
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x

        # ☓forward時に(N, T, D)の形状を出力しているからbackwardでもdoutは(N, T, D)で逆伝搬する
        # ☓したがって、backward時は入力値のdoutの次元をif文で調べなくて良い
        # 上記説明が間違っている理由、TimeAffineの次にPoolingのようなレイヤーがあるとdoutはforward時の出力形状とは限らないから
        # N, T, D = dout.shape
        # 形状を取得するときは、次の層から逆伝搬した値ではなく、絶対的に変化しないforwardの入力値xを使用する

        # 形状の取得方法を入力値の次元によって変更(TimeAffineをレイヤーの最初にする場合とそれ以外の場合で入力値の次元はことなるから)
        if x.ndim == 3:
            N, T, D = x.shape
            W, b = self.params
        
        elif x.ndim == 2:
            N, T = x.shape
            W, b = self.params
            _, D = W.shape

        """元の計算方法

        # dout = dout.reshape(N*T, -1)
        # rx = x.reshape(N*T, -1)

        # db = np.sum(dout, axis=0)
        # dW = np.dot(rx.T, dout)
        # dx = np.dot(dout, W.T)
        # dx = dx.reshape(*x.shape)

        # self.grads[0][...] = dW
        # self.grads[1][...] = db

        # return dx
        """

        # 形状をreshape前のoutの形に戻す(N * T, -1)
        dout = dout.reshape(N * T, -1)
        # dWを計算するためにxの形状を変更(N * T, -1)
        rx = x.reshape(N * T, -1)

        # dWの計算
        dW = np.dot(rx.T, dout)
        # dbの計算
        db = np.sum(dout, axis=0)
        # dxの計算
        dx = np.dot(dout, W.T)
        # 形状をxと同じ状態に戻す(xの次元が変化しても柔軟に対応できるようにするため*x.shapeとする)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


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
        ls = np.log(ys[np.arange(N * T), ts])
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
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx

class TimeReLUWithLoss:
    def __init__(self, lam=1):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1
        self.lam = lam
    
    def forward(self, xs, ts, Wx):
        N, T, D = xs.shape

        # 時系列をまとめる
        xs = xs.reshape(N * T, D)
        ts = ts.reshape(N * T, D)

        # ReLU(0以下の値を0に変更)
        ys = np.maximum(0, xs)

        # MSE
        diff = ys - ts
        loss = 0.5 * np.sum(diff ** 2)
        loss /= (N * T)

        # WxでL2正則化
        loss += 0.5 * self.lam * np.sum(Wx ** 2)

        self.cache = (xs, ys, ts, diff, (N, T, D))

        return loss
    
    def backward(self, dout=1):
        xs, ys, ts, diff, (N, T, D) = self.cache

        # MSEの微分
        dy = diff / (N * T)
        dy *= dout

        # ReLUの微分
        dx = dy.copy() # 値渡し
        dx[xs <= 0] = 0

        # 元の形に戻す
        dx = dx.reshape(N, T, D)

        return dx