import sys
sys.path.append("..")
import common.config
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq, to_gpu
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

GPU = common.config.GPU

# データセット読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
char_to_id, id_to_char = sequence.get_vocab()

train_data_size = int(len(x_train) * 0.3)
test_data_size = int(len(x_test) * 0.3)

print(train_data_size, test_data_size)

x_train, t_train = x_train[:train_data_size], t_train[:train_data_size]
x_test, t_test = x_test[:test_data_size], t_test[:test_data_size]

# --- 追加すべき処理 ---
if GPU:
    x_train, t_train = to_gpu(x_train), to_gpu(t_train)
    x_test, t_test = to_gpu(x_test), to_gpu(t_test)
# --------------------

file_name = "first-test.png"

is_reverse = False

if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

    file_name = "second-test.png"
# print(x_train.shape)

# print(train_data_size)
# print(test_data_size)

# print("".join([id_to_char[i] for i in x_train[-1]]))
# print("".join([id_to_char[i] for i in t_train[-1]]))

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# モデル/オプティマイザ/トレーナーの生成
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
model = Seq2seq(vocab_size, wordvec_size, hidden_size)
file_name = "peeky-test.png"

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        # verbose = i < 10
        verbose = i >= 0
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print(f"val acc {acc*100:.3f}")


plt.plot(np.arange(len(acc_list)), acc_list, marker="o", color="orange")
plt.savefig(file_name)