import sys
sys.path.append("..")
from dataset import sequence

(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt", seed=1984)
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

print("".join([id_to_char[c] for c in x_train[0]]))
print("".join([id_to_char[c] for c in t_train[0]]))

x_train = x_train[:, ::-1]

print("".join([id_to_char[c] for c in x_train[0]]))
print("".join([id_to_char[c] for c in t_train[0]]))


# print(x_train[0])
# print(t_train[0])

# print("".join(id_to_char[c] for c in x_train[0]))
# print("".join(id_to_char[c] for c in t_train[0]))

# for t1, t2 in zip(t_test[:5, :-1], t_test[:5, 1:]):
#     print("".join(id_to_char[c] for c in t1), len(t1))
#     print("".join(id_to_char[c] for c in t2), len(t2))

# print(len(t_train[0]))
# print(char_to_id)