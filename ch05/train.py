import sys
sys.path.append("..")

from common.optimizer import SGD
from dataset import ptb
from ch05.simple_rnnlm import SimpleRnnlm
from common.trainer import RnnlmTrainer

# ハイパーパラメータ
batch_size = 10
wordvec_size = 100
hidden_size = 100 # RNNの隠れ状態ベクトルの要素数
time_size = 5 # Trucated BPTTの展開する時間サイズ
lr = 0.1
max_epoch = 1000

# 学習データの読み込み(データセットを小さくする)
corpus, word_to_id, id_to_word = ptb.load_data("train")

corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1] # 入力
ts = corpus[1:] # 出力
data_size = len(xs)

print(f"corpus size: {corpus_size}, vocabulary size: {vocab_size}")

# モデルの生成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD()

trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch=max_epoch, batch_size=batch_size, time_size=time_size)

filename = "train_result.png"
trainer.plot(filename=filename)