from curses import window
import sys
sys.path.append("..")
import numpy as np
from common.util import create_contexts_target, preprocess
from cbow import CBOW
from common.trainer import Trainer
from common.optimizer import Adam
import pickle
from dataset import ptb

def main():
    window_size = 5
    H = 100 # 各単語の分散ベクトルの次元数
    sample_size = 3 # ネガティブサンプリングする個数
    power = 0.75

    batch_size = 100
    max_epoch = 2

    corpus, word_to_id, id_to_word = ptb.load_data("train")
    contexts, targets = create_contexts_target(corpus, window_size) 

    vocab_size = len(word_to_id)

    filename = "ptb_sample.png" # 実行結果
    pkl_filename = "ptb_params.pkl"

    optimizer = Adam()
    model = CBOW(corpus, vocab_size, H, window_size, sample_size, power)

    trainer = Trainer(model, optimizer)
    trainer.fit(contexts, targets, batch_size=batch_size, max_epoch=max_epoch)

    trainer.plot(filename)

    # 分散表現とその他付属情報をを保存
    params = {
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "word_vec": trainer.model.word_vec.astype(np.float16)
    }

    with open(pkl_filename, "wb") as f:
        pickle.dump(params, f, -1)

if __name__ == "__main__":
    main()