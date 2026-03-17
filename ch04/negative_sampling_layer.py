import sys
sys.path.append("..")
from common.layers import *
import numpy as np
import collections
from common.np import *

# CBOW の出力層で使う「内積を計算する層」
# h (文脈ベクトル) と、ターゲット語の埋め込みベクトルの内積を計算する
class EmbeddingDot:
    def __init__(self, W) -> None:
        # Embedding 層を内部に持つ (単語 ID に対応する埋め込みベクトルを取り出す)
        self.embed = Embedding(W)

        # パラメータと勾配をまとめて持つ (他の層と同じインターフェース)
        self.params = self.embed.params
        self.grads = self.embed.grads

        # 順伝播で使った値を保存しておくためのキャッシュ
        self.cache = None

    def forward(self, h, idx):
        # ターゲット単語 ID (idx) に対応する埋め込みベクトルを取り出す
        target_W = self.embed.forward(idx)

        # h と target_W の内積を計算
        # (各バッチごとに要素積をとり、埋め込み次元方向(行方向)に和をとる)
        out = np.sum(target_W * h, axis=1)

        # 逆伝播で必要になるので保存
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        # forward で使った値を取り出す
        h, target_W = self.cache

        # dout: (バッチサイズ,) → (バッチサイズ,1) に形を変換
        # (この後のブロードキャストを正しくするため)
        dout = dout.reshape(dout.shape[0], 1)

        # target_W に対する勾配
        # 内積 out = sum(h * target_W) に対する dW = dout * h
        dtarget_W = dout * h

        # 埋め込み行列 W のうち、該当する行だけに勾配を反映する
        self.embed.backward(dtarget_W)

        # 入力ベクトル h に対する勾配
        # dh = dout * target_W
        dh = dout * target_W

        return dh


# ネガティブサンプリングを行う
class UnigramSmapler:
    # 単語の確率分布を取得
    def __init__(self, corpus, power, sample_size) -> None:
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter(corpus)

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(self.vocab_size)

        for i in range(self.vocab_size):
            self.word_p[i] = counts[i]
        
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
    
    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy() # copy.deepcopyとほぼ同じ(データを新しいメモリにコピーする)
                target_idx = target[i]
                p[target_idx] = 0 # ターゲットを負の例としてサンプリングしないようにターゲットの確率を0にする
                p /= p.sum()

                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）で計算するときは、速度を優先
            # 負例にターゲットが含まれるケースがある
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample
    

# 出力層を効率的に計算するための層(EmbeddingDotとUnigramSmaplerを使用)
class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5) -> None:
        self.sample_size = sample_size
        self.sampler = UnigramSmapler(corpus, power, self.sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
                

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例のフォワード
        target_label = np.ones(batch_size, dtype=np.int32) # 正解ラベル
        target_score = self.embed_dot_layers[0].forward(h, target) # 正解ラベルに対する予測
        loss = self.loss_layers[0].forward(target_score, target_label)

        # 負例のフォワード
        negative_label = np.zeros(batch_size, dtype=np.int32)
        # embedとlossレイヤーをiで指定するために、iを1から開始する。
        for i in range(1, self.sample_size + 1):
            negative_target = negative_sample[:, i - 1] # ネガティブサンプルのshapeは(batch_size. sample_size)。そのままiを使うと添字エラーになってしまう。
            negative_score = self.embed_dot_layers[i].forward(h, negative_target)
            loss += self.loss_layers[i].forward(negative_score, negative_label)
        
        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        
        return dh
