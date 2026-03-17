import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
# from ch05.simple_rnnlm import SimpleRnnlm


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

# 学習時に使用する変数
max_iters = data_size // (batch_size * time_size) # 1エポックの中で重みの更新を行う回数(1000文字の巻物を10人で5文字ずつ読み進めていく考え方)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# モデルの生成(Timeレイヤーは単語数, 単語埋め込み次元, 状態hの次元の単位で学習)
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# ミニバッチの各サンプルの読み込み開始位置を計算
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # ミニバッチの取得
        batch_x = np.empty((batch_size, time_size), dtype="i") # (10, 5)
        batch_t = np.empty((batch_size, time_size), dtype="i") # (10, 5)
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        # 勾配を求め、パラメータを更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
    
    # エポックごとにパープレキシティの評価
    ppl = np.exp(total_loss / loss_count)
    print(f"epoch {epoch} perplexity {ppl}")
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 学習結果の描画
plt.plot(np.arange(max_epoch), ppl_list)
plt.savefig("rnnlm_train_result.png")