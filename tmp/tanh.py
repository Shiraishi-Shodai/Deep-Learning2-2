import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib

def tanh(x):
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y

# X = np.arange(-50, 50, 0.001)
# y = tanh(X)

# plt.plot(X, y)
# plt.title("tanh関数")
# plt.grid()
# plt.show()

print(np.exp(2), np.exp(1), np.exp(-1), np.exp(0), np.exp(-2))
print(1 * (1 / np.exp(1)))
print(1 * (1 / (np.exp(1) * np.exp(1))))

