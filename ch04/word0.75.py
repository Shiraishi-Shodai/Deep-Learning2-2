import numpy as np
import matplotlib.pyplot as plt

# 語彙サイズ（例として5000語）
V = 5000
# ランク（1 = 最頻出語）
ranks = np.arange(1, V + 1)

# Zipf分布で出現確率を擬似的に生成
s = 1.0  # Zipf指数
freq = 1.0 / (ranks ** s)
p = freq / freq.sum()  # 正規化して確率分布にする (a=1.0 の場合)

# a=0.75 を適用
a = 0.75
q = p ** a
q = q / q.sum()  # 正規化

# プロット
plt.figure(figsize=(7,5))
plt.plot(ranks, p, label="a = 1.0 (baseline)")
plt.plot(ranks, q, label="a = 0.75 (negative sampling)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Rank (1 = most frequent)")
plt.ylabel("Probability")
plt.title("Effect of exponent a=0.75 on negative-sampling distribution")
plt.legend()
plt.tight_layout()
plt.show()