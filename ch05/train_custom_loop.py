import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# ハイパーパラメータ
batch_size = 10
wordvec_size = 100
hidden_size = 100 # RNNの隠れ状態ベクトルの要素数
time_size = 5 # Trucated BPTTの展開する時間サイズ
lr = 0.1
max_epoch = 100

# 学習データの読み込み(データセットを小さくする)
corpus, word_to_id, id_to_word = ptb.load_data("train")

corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1] # 入力
ts = corpus[1:] # 出力
data_size = len(xs)

print(f"corpus size: {corpus_size}, vocabulary size: {vocab_size}")

# 学習時に使用する変数
max_iters = data_size // (batch_size * time_size)
time_idx = 0 # ミニバッチデータを取得する際に、時系列方向に指定するインデックスをずらす役割
total_loss = 0 # 各ミニバッチを使った予測時の合計損失関数
loss_count = 0 # 損失関数を計算した回数
ppl_list = [] # perplexity list

print(f"data size: {data_size}, max iters: {max_iters}")

# モデルの生成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD()

# ミニバッチの各サンプルの読み込み開始位置を計算(配列はインデックスが0始まりのため、corpus_sizeから1を引く)
jump = (corpus_size - 1) // batch_size
print(f"jump: {jump}")

offsets = [i * jump for i in range(batch_size)]
print(f"offsets: {offsets}")


for epoch in range(1):
    # max_iter回forを回すことですべてのデータを使用する。
    for iter in range(1):
        # ミニバッチの取得(バッチサイズ, タイムサイズ)
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")
        
        # バッチデータの列インデックス
        for t in range(time_size):
            # バッチデータの行インデックス
            for i, offset in enumerate(offsets):
                # xs, tsのインデックス範囲を超えたら最初に戻るようにdata_sizeの余りを計算している
                batch_x[i, t] = xs[(offset + time_idx) % data_size] 
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        # 勾配を求め、パラメータ更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # エポックごとにperplexityを評価
    ppl = np.exp(total_loss / loss_count)
    print(f"| epoch: {epoch + 1} | perplexity: {ppl}")
    ppl_list.append(ppl)
    total_loss, loss_count = 0, 0