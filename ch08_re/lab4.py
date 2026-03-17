from attention_layer import *
from attention_seq2seq import *

V, D, H = 2, 3, 4
ad = AttentionDecoder(V, D, H)
print(ad.attention.params)
print(ad.params)