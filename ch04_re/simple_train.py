from curses import window
import sys
sys.path.append("..")
import numpy as np
from common.util import create_contexts_target, preprocess
from cbow import CBOW
from common.trainer import Trainer
from common.optimizer import Adam
import pickle

def main():
    text = "You say goodby and I say hell."
    window_size = 1

    corpus, word_to_id, id_to_word = preprocess(text)
    contexts, targets = create_contexts_target(corpus, window_size) 

    print(type(contexts), type(targets))

    vocab_size = len(word_to_id)
    H = 3 # 各単語の分散ベクトルの次元数
    sample_size = 3 # ネガティブサンプリングする個数
    power = 0.75

    batch_size = 3
    max_epoch = 1000

    filename = "simple_sample.png" # 実行結果
    model_name = "simple_sample_model.pkl"

    optimizer = Adam()
    model = CBOW(corpus, vocab_size, H, window_size, sample_size, power)

    trainer = Trainer(model, optimizer)
    trainer.fit(contexts, targets, batch_size=batch_size, max_epoch=max_epoch)

    trainer.plot(filename)

    # モデルを保存
    with open(model_name, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()