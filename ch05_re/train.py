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
from common.optimizer import SGD
from dataset import ptb

def main():

    # ハイパーパラメータの設定
    batch_size = 10
    wordvec_size = 100 # 埋め込みベクトルの次元数
    hidden_size = 100 # RNNの隠れ状態ベクトルの要素数(次元数)
    time_size = 5 # Truncated BPTTの展開する時間サイズ
    lr = 0.1
    max_epoch = 100 # データ全体を学習に何周使用するか

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1] # 入力
    ts = corpus[1:] # 教師ラベル
    data_size = len(xs)
    print(f"curpus size: {corpus_size}, vocabulary size: {vocab_size}")

    filename = "sample_train_result.png"

    # 学習時に使用する変数
    max_iters = data_size // (batch_size * time_size) # 1エポックの中で重みの更新を行う回数(1000文字の巻物を10人で5文字ずつ読み進めていく考え方)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []

    # モデルの生成(Timeレイヤーは単語数, 単語埋め込み次元, 状態hの次元の単位で学習)
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RNNTrainer(model, optimizer)

    # 学習
    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot(filename)

    # テスト
    pred_xs, pred_loss = trainer.predict(xs, ts, batch_size, time_size)
    print(pred_xs)
    print(pred_loss)

if __name__ == "__main__":
    main()