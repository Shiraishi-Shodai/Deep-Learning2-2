import numpy as np
import matplotlib.pyplot as plt
import math
import japanize_matplotlib

# x 軸の値を用意
x = np.linspace(-2, 2, 400)

# e^x とその増え方（微分）
exp_x = np.exp(x)
d_exp_x = np.exp(x)  # e^x は微分しても e^x のまま

# 2^x とその増え方（微分）
pow2_x = 2 ** x
d_pow2_x = math.log(2) * (2 ** x)  # 増え方が自分自身と一致しない

# グラフを描画
plt.figure()

# e^x（実線）とその増え方（破線）
plt.plot(x, exp_x, label="e^x")
plt.plot(x, d_exp_x, linestyle="dashed", label="d/dx e^x")

# 2^x（実線）とその増え方（破線）
plt.plot(x, pow2_x, label="2^x")
plt.plot(x, d_pow2_x, linestyle="dashed", label="d/dx 2^x")

plt.title("『増え方が自分自身と一致する』のは e^x だけ")
plt.xlabel("x")
plt.ylabel("値")
plt.legend()

plt.show()

# print(math.log(math.e**2))
# print(math.log(2))