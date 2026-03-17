# coding: utf-8
from pickletools import optimize
import sys

sys.path.append('..')
import numpy
import time
import matplotlib.pyplot as plt
from common.np import *  # import numpy as np
from common.util import clip_grads
from my_util.functions import MSE

class RNNTrainer:
    def __init__(self, model, optimizer) -> None:
        self.optimizer = optimizer
        self.model = model
        self.time_idx = None
        self.evaluation_list = None # perplexity list
        self.eval_interval = None
        self.current_epoch = 0
        self.current_idx = 0
        self.predict_list = None
        self.loss_list = None
        self.l2_norm_list = None
    
    def get_batch(self, x, t, batch_size, time_size):

        batch_x = np.empty((batch_size, time_size)).astype("f")
        batch_t = np.empty((batch_size, time_size)).astype("f")

        data_size = len(x)
        jump = (data_size - 1) // batch_size
        offsets = [jump * i for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            
            self.time_idx += 1

        return batch_x, batch_t
    
    def get_test_batch(self, x, t, batch_size, time_size):
        """
        わりきれなかったデータは捨てる。
        """
        data_size = x.shape[0]
        batch_x = np.zeros((batch_size, time_size))
        batch_t = np.zeros((batch_size, time_size))
        x = x.flatten()

        for b in range(batch_size):
            for i in range(time_size):
                batch_x[b, i] = x[self.current_idx + i]
                batch_t[b, i] = t[self.current_idx + i]
            self.current_idx += time_size
    
        return batch_x, batch_t
    
    def fit(self, xs, ts, max_epoch=100, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size) 
        self.time_idx = 0
        self.evaluation_list = []
        self.loss_list = []
        self.l2_norm_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss, loss_count = 0, 0

        start_time = time.time()

        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 勾配を求め、パラメータ更新
                loss = model.forward(batch_x, batch_t)
                model.backward()

                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    elapsed_time = time.time() - start_time
                    l2_norm = get_l2_norm(model.params)
                    print('| epoch %d |  iter %d / %d | time %d[s] | MSE %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, loss))
                    self.loss_list.append(float(loss))
                    self.l2_norm_list.append(float(l2_norm))
                    # self.evaluation_list.append(float(mse))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1
    
    def predict(self, X_test, T_test, batch_size=20, time_size=35):
        self.current_idx = 0
        self.predict_list = np.zeros_like(T_test)
        data_size = X_test.shape[0]
        max_iters = data_size // (batch_size * time_size)
        surplus_data_size = data_size % (batch_size * time_size)
        T_prev = np.zeros_like(T_test)
        idx = 0

        for iters in range(1, max_iters + 1):
            batch_x, batch_t = self.get_test_batch(X_test, T_test, batch_size, time_size)
            T_prev[idx : idx + (batch_size * time_size)] = self.model.predict(batch_x).flatten()
            idx += batch_size * time_size
        return T_prev

    def plot(self, filename, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations')
        plt.ylabel('evalution')
        plt.savefig(filename)
        plt.clf()

        x = np.arange(len(self.l2_norm_list))
        plt.plot(x, self.l2_norm_list)
        plt.xlabel("iterations")
        plt.ylabel("l2 norm")
        plt.savefig("l2_norm.png")
        plt.clf()

    def test_evalution_plot(self, x, t, t_pred, filename, ylim=None):
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, t, label='original', c="red")
        plt.plot(x, t_pred, label='predict', c="blue")
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend()
        plt.savefig(filename)

def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

def get_l2_norm(grads):
    l2_norm = 0
    for i, grad in enumerate(grads):
        norm = np.sum(grad ** 2)
        l2_norm += norm
    
    l2_norm = np.sqrt(l2_norm)
    return l2_norm