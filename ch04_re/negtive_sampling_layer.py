import sys
from typing import Collection
sys.path.append("..")
from common.layers import Embedding, SigmoidWithLoss
import numpy as np
from collections import Counter

class EmbeddingDot:
    def __init__(self, W) -> None:
        self.embed = Embedding(W)

        # 追加
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
    
    def forward(self, h, index):
        target_W = self.embed.forward(index)
        out = np.sum(h * target_W, axis=1)

        self.cache = (h, target_W)

        return out
    
    def backward(self, dout):
        h, target_W = self.cache

        dout = dout.reshape(-1, 1)

        dtarget_W = dout * h # 列方向にdoutをhに掛ける

        self.embed.backward(dtarget_W)

        dh = dout * target_W

        return dh


# ネガティブサンプリングを行う
class UnigramSmapler:
    def __init__(self, corpus, sample_size=5, power=0.75) -> None:
        self.corpus = corpus
        self.sample_size = sample_size

        counter = Counter(corpus)
        self.vocab_size = len(counter)

        self.word_p = np.zeros(self.vocab_size)
        self.word_p = np.power(list(counter.values()), power, dtype=np.float32)
        self.word_p /= np.sum(self.word_p)

    
    def get_negative_sampling(self, target):

        batch_size = target.shape[0]

        # 返却値を0で初期化
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for b in range(batch_size):
            # targetをネガティブサンプリングで取得しないようにする。
            p = self.word_p.copy()
            p[target] = 0
            p /= np.sum(p)

            negative_sample[b] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)      

        return negative_sample  


# 出力層の定義
class NegativeSamplingLoss:

    # EmbeddingDotとUnigramSmaplerを正解例 + 負例の数用意
    def __init__(self, W, corpus, sample_size=5, power=0.75) -> None:

        self.sample_size = sample_size

        self.embed_dots = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.unigram_sampler = UnigramSmapler(corpus, sample_size, power)
    
        self.loss = 0

        self.params, self.grads = [], []

        for layer in self.embed_dots:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):

        batch_size = target.shape[0]

        # 正例と負例の正解データを用意
        correct_label = np.ones(batch_size, dtype=np.int32)
        negative_label = np.zeros(batch_size, dtype=np.int32)

        # ネガティブサンプルを取得
        negative_samples = self.unigram_sampler.get_negative_sampling(target)

        # 正例データの予測と損失計算
        score = self.embed_dots[0].forward(h, target)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # 負例データの予測と損失計算
        for i in range(1, self.sample_size + 1):
            negative_target = negative_samples[:, i - 1]
            score = self.embed_dots[i].forward(h, negative_target)
            loss += self.loss_layers[i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):

        dh = 0
        
        for i in range(self.sample_size + 1):
            dscore = self.loss_layers[i].backward(dout)
            dh += self.embed_dots[i].backward(dscore)
        
        return dh