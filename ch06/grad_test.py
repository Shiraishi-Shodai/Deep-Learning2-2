import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib

# tanhとtanhの勾配(tanhの微分はxが0のとき1になりxが0から離れるとyは0に近くなる)
tanh_x = np.linspace(-5, 5, 50)
tanh_y = np.tanh(tanh_x)

grad_tanh_x = tanh_x
grad_tanh_y = 1 - (np.tanh(grad_tanh_x)**2)

plt.plot(tanh_x, tanh_y, c="red", label="tanh(x)")
plt.plot(grad_tanh_x, grad_tanh_y, c="orange", label="dy/dx")
plt.legend()
plt.show()

# 行列の微分(同じ重みで繰り返し乗算するため勾配消失や勾配爆発が起こる)

# N = 2
# H = 3
# T = 20 

# dh = np.ones((N, H))
# np.random.seed(3)
# Wh = np.random.randn(H, H) # before
# # Wh = np.random.randn(H, H) * 0.5 # after
# print("重み")
# print(Wh)

# norm_list = []
# for t in range(T):
#     dh = np.dot(dh, Wh.T)
#     print(f"{T}回目")
#     print(dh)
#     norm = np.sqrt(np.sum(dh**2)) / N
#     norm_list.append(norm)

# plt.plot(np.arange(T), norm_list)
# plt.show()