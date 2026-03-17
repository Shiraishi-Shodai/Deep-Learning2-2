# coding: utf-8
import sys
sys.path.append('..')
import os
from common.np import *
import numpy as np

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)

    # 勾配にかける値が1以下(勾配のL2ノルムが閾値より大きいとき)
    # 勾配クリップを行う
    if rate < 1:
        for grad in grads:
            grad *= rate

def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    """パープレキシティの計算
    データをバッチごとに分けて各バッチで求めた損失の合計と損失を計算した
    値を使用
    """
    print("パープレキシティを評価")
    corpus_size = len(corpus)
    total_loss = 0
    max_iters (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
    
        for t in range(time_size):
            for i in range(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]
    
        try:
            loss = model.forward(xs, ts, train_flg=Flase)
        except TypeError:
            loss = model.forward(xs, ts)
        
        total_loss += loss
    
        sys.stdout.write("\r%d / %d" % (iters, max_iters))
        sys.stdout.flush()
    print("")
    ppl = np.exp(total_loss / max_iters)
    return ppl

