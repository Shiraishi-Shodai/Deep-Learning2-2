import sys
sys.path.append("..")
from common.layers import Softmax
import numpy as np

N, T, H = 3, 2, 3
hs = np.random.randn(N, T, H).astype("i8")
h = np.random.randn(N, H).astype("i8")

hr = h.reshape(N, 1, H).repeat(T, axis=1)

t = hs * hr

s = np.sum(t, axis=2)

print(hr, end="\n")
print(s)