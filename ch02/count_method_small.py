import sys
sys.path.append("..")
import re
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

np.set_printoptions(precision=3)

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)
# print(corpus, word_to_id, id_to_word)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size, window_size=1)
# print(C[word_to_id["you"]])

c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]

# print(cos_similarity(c0, c1))

# most_similar(id_to_word[0], word_to_id, id_to_word, C)

print(C)
W = ppmi(C)
print(W)