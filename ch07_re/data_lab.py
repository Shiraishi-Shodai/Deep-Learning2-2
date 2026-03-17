import sys
sys.path.append("..")
import numpy as np
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
from common.base_model import BaseModel
from seq2seq import Encoder
from dataset import sequence

def char_decode(id_list, id_to_char):
    decode_value = "".join([id_to_char[id] for id in id_list])
    return decode_value

# データセット読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
char_to_id, id_to_char = sequence.get_vocab()

x = np.array([x_train[0]])
t = np.array([t_train[0]])
# x = x_train[:2, :]
# t = t_train[:2, :]

data_size = len(x_train)
batch_size = 128
wordvec_size = 5
vocab_size = len(id_to_char)
rn = np.random.randn

max_iters = data_size // batch_size

# モデルの部品
embed_W = (rn(vocab_size, wordvec_size) / 100).astype("f")
time_embedding = TimeEmbedding(embed_W)

rng = np.random.default_rng()

for iters in range(max_iters):
    """データの確認"""
    batch_x = x[iters*batch_size:(iters+1)*batch_size]
    batch_t = t[iters*batch_size:(iters+1)*batch_size]
    print("入力データ", end="\n")
    print(batch_x, end="\n")
    print(batch_t, end="\n")
    print(batch_x.shape, batch_t.shape)

    N, T, D = len(batch_x), batch_x.shape[1], wordvec_size

    for i, (xi, ti) in enumerate(zip(batch_x, batch_t)):
        shiki = char_decode(xi, id_to_char)
        answer = char_decode(ti, id_to_char)

        print(f"問{i + 1}： {shiki} = {answer}")
        
        print(f"Decoderの入力と出力")
        print(f"Decoderの入力")
        decoder_input = char_decode(ti[:-1], id_to_char)
        print(decoder_input)
        print(f"Decoderの出力")
        decoder_output = char_decode(ti[1:], id_to_char)
        print(decoder_output)
    
    """Embeddingの動作の確認(forward編)"""
    # print(f"vocab_size: {vocab_size}")
    # embed_output = time_embedding.forward(batch_x)
    # print("Embeddingの出力", end="\n")
    # print(embed_output, end="\n")
    # print(f"embdedding size: {embed_output.shape}")

    """Embeddingの動作の確認(backward編)"""
    # embed_dout = rng.integers(1, 10, size=(N, T, D), endpoint=True)
    # print(f"dout: {embed_dout}")
    # print(f"dout size: {embed_dout.shape}")

    # time_embedding.backward(embed_dout)



    
    break




