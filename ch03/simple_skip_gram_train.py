import sys
sys.path.append("..")
from simple_skip_gram import SimpleSkipGram
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import create_contexts_target, convert_one_hot, preprocess

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 100

text = 'You say goodbye and I say hello .'

corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id.keys())

contexts, target = create_contexts_target(corpus, window_size)

contexts = convert_one_hot(contexts, vocab_size)
target = convert_one_hot(target, vocab_size)

model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(target, contexts, max_epoch, batch_size)
trainer.plot()

word_vec = trainer.model.word_vec

for word, word_id in word_to_id.items():
    print(f"{word}: {word_vec[word_id]}")


