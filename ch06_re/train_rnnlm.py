import sys
sys.path.append("..")
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm

# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 100 # 単語ベクトルの次元
hidden_size = 100 # 隠れ状態hの次元数
time_size = 35 # 一度に扱う時刻の数
lr = 20.0
max_epoch = 1
max_grad = 0.25 # 勾配クリップを行う閾値

# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_test, _, _ = ptb.load_data("test")
vocab_size = len(word_to_id) # 単語数
xs = corpus[:-1]
ts = corpus[1:]

# モデルの生成
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# # 勾配クリピングを適用して学習
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# テストデータで評価(LSTMの隠れ状態hと記憶セルcをリセット後)
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print(f"test perplexity: {ppl_test}")

# # パラメータの保存
model.save_params()