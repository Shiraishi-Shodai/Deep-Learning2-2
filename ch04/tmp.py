import numpy as np

words = ["you", "say", "goodbye", "I", "hello", "."]
print(np.random.choice(words, size=5))
print(np.random.choice(["I", "I"], size=2, replace=False))

a = dict.fromkeys(list(["I", "I"]), "seen")
print(a)