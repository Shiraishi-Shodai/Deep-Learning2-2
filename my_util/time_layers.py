import numpy as np

from common.layers import Embedding
from common.functions import softmax, sigmoid


class LSTM:
    def __init__(self, Wx, Wh, b):
        # ここでのWx, Wh, bは各ゲートの重みを並べたもの
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        # 各ゲートの出力を求める A = (N, H)
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # 各ゲートの出力を取り出す
        f = A[:, :H]
        g = A[:, H : H * 2]
        i = A[:, H * 2 : H * 3]
        o = A[:, H * 3 :]

        # 各ゲートの活性化関数通過後の値を求める
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        # 記憶セルを計算
        c_next = (c_prev * f) + (g * i)

        # 次の隠れ状態hを計算
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)

        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        # セルcのforward時におけるforegetゲート通過後の勾配を計算
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        # セルcのforgetゲート通過前の勾配を計算
        dc_prev = ds * f

        # forward時における活性化関数通過後のf ~ oゲートの出力の勾配を求める
        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        # forward時における活性化関数通過前のf ~ oゲートの勾配を求める
        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        # forward時に分配した重みやバイアスを結合し一気に勾配を計算
        dA = np.hstack((df, dg, di, do))

        dWx = np.dot(x.T, dA)
        dWh = np.dot(h_prev.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful
    
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h  is None:
            self.h = np.zeros((N, H), dtype="f")
        
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype="f")
        
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)
        
        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype="f")
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # repeatがあるためdhを合計
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh
        
        return dxs
    
    def set_state(self, h, c=None):
        self.h, self.c = h, c
    
    def reset_state(self):
        self.h, self.c = None, None


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        
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
        dWx = np.dot(x.T, dt)
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
        return out.reshape(N, T, D)

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

        # 形状をreshape前のoutの形に戻す(N * T, 1)
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

    
class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask


class TimeReLUWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1
    
    def forward(self, xs, ts):
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
