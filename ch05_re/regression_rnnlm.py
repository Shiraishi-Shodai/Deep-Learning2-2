import sys
sys.path.append("..")
import numpy as np
from ch05_re.time_layers import *
import pickle

class RNNRegressor:
    def __init__(self, input_size=1, vec_size=100, hidden_size=50, lam=1):
        I, D, H = input_size, vec_size, hidden_size
        rn = np.random.randn

        # TODO: マジックナンバー1の変更
        scaleAffin1_W = np.sqrt(2 / I)
        affine1_W = scaleAffin1_W * (rn(I, D) / 100).astype("f")
        affine1_b = np.zeros(D).astype("f")

        scaleWx = np.sqrt(1 / D).astype("f")
        scaleWh = np.sqrt(1 / H).astype("f")
        rnn_Wx = scaleWx * (rn(D, H)).astype("f")
        rnn_Wh = scaleWh * (rn(H, H) / np.sqrt(H)).astype("f")
        rnn_b = np.zeros(H).astype("f")

        scaleAffin2_W = np.sqrt(2 / H)
        affine2_W = scaleAffin2_W * (rn(H, I) / np.sqrt(H)).astype("f")
        affine2_b = np.zeros(I).astype("f")

        self.layers = [
            TimeAffine(affine1_W, affine1_b),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b),
            TimeAffine(affine2_W, affine2_b)
        ]

        # 出力層の活性化関数と損失関数の設定: (ReLU, MSE)
        self.loss_layer = TimeReLUWithLoss()
        self.rnn_layer = [
            self.layers[1]
        ]

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs):
        out = xs
        for layer in self.layers:
            out = layer.forward(out)
        return out
        
    def forward(self, xs, ts):
        out = self.predict(xs)
        Wx = self.rnn_layer[0].params[0]
        loss = self.loss_layer.forward(out, ts, Wx)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout
    
    def reset_state(self):
        """隠れ状態hをNoneにする
        """
        for rnn_layer in self.lstm_layers:
            rnn_layer.reset_state()
    
    def save_params(self, file_name="RNNRegressor.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_name="RNNRegressor.pkl"):
        with open(file_name, "rb") as f:
            self.params = pickle.load(f)