import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.trainer import Trainer
from common.util import preprocess, create_contexts_target, convert_one_hot
import numpy as np

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 100

text = "You say goodbye and I say hello."
corpus , word_to_id, id_to_word = preprocess(text) # (8,)(6,)(6,)
contexts, target = create_contexts_target(corpus, window_size=1) # (6, 2), (6,)

contexts = convert_one_hot(contexts, len(word_to_id.keys()))
target = convert_one_hot(target, len(word_to_id.keys()))

model = SimpleCBOW(len(word_to_id.keys()), hidden_size)
optimizer = Adam()

trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vec  = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vec[word_id])