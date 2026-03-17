# coding: utf-8
from pickletools import optimize
import sys
sys.path.append('..')
import numpy
import time
import matplotlib.pyplot as plt
from common.np import *  # import numpy as np
from trainer import RNNTrainer
from simple_rnnlm import SimpleRnnlm
from common.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam
from dataset import ptb
import pandas as pd
from common.util import custom_preprocess
from regression_rnnlm import RNNRegressor
from my_util.functions import MSE
from sklearn.preprocessing import StandardScaler

def main():

    # データの用意 (1000, 2)
    df = pd.read_csv("../data/sin_data.csv")

    train_X = df["X"].to_numpy(dtype="f").reshape(-1, 1)
    train_T = df["y"].to_numpy(dtype="f")

    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)

    # ハイパーパラメータの設定
    batch_size = 20
    time_size = 30
    lr = 0.0011
    momentum = 0.5
    decay_rate = 0.9
    beta1 = 0.1
    beta2 = 0.4
    max_epoch = 10000
    max_grad = 100
    eval_interval = 20
    lam = 0

    input_size = train_X.shape[1]
    vec_size = 100
    hidden_size = 5

    model = RNNRegressor(input_size, vec_size, hidden_size, lam)
    # optimizer = SGD(lr)
    # optimizer = Momentum(lr, momentum)
    # optimizer = Nesterov(lr, momentum)
    # optimizer = AdaGrad(lr)
    # optimizer = RMSprop(lr, decay_rate)
    optimizer = Adam(lr, beta1, beta2)

    trainer = RNNTrainer(model, optimizer)
    trainer.fit(train_X, train_T, max_epoch, batch_size, time_size, max_grad, eval_interval)

    file_name = "train_mse"
    trainer.plot(file_name)

    # テスト結果
    test_file_name = "test.png"
    pred_T = trainer.predict(train_X, train_T, batch_size, time_size)
    trainer.test_evalution_plot(train_X, train_T, pred_T, test_file_name)
    mse = MSE(train_T, pred_T)
    print(f"平均二乗誤差 {mse}")

if __name__ == "__main__":
    main()