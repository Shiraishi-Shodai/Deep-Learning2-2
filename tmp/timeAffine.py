import numpy as np

# TimeAffineのforwardテスト(Xの形状が(N, T)のとき)
N = 2
T = 3
D = 4

x = np.arange(1, 7).reshape(N, T)
u = x.reshape(N * T, -1)
w = np.linspace(10, 40, D).reshape(1, -1)

u = np.dot(u, w)
z = u.reshape(N, T, D)
# print(z)

# TimeAffineのbackwardテスト(Xの形状が(N, T)のとき)
dout = z # 逆伝搬で前のレイヤから流れて来たデータと仮定する

# ①　(N * T, 1) dx, dw, dbを計算するためにforward時のuの形状に戻す
dout = dout.reshape(N * T, -1) 

# ② dxの計算
dx = np.dot(dout, w.T)
dx = dx.reshape(*x.shape)
# print(w.T)
# print(dx)

# ③ dwの計算
# print(x.reshape(N * T, -1).T)
dw = np.dot(x.reshape(N * T, -1).T, dout)
# print(dw)

# ④ dbの計算
db = np.sum(dout, axis=0)
# print(db)