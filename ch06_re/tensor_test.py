import numpy as np

x = np.arange(6).reshape(2, 3)
H = 2

np.random.seed(3)

Wf = np.random.randint(1, 10, 6).reshape(3, 2)
Wg = np.random.randint(1, 10, 6).reshape(3, 2)
Wi = np.random.randint(1, 10, 6).reshape(3, 2)
Wo = np.random.randint(1, 10, 6).reshape(3, 2)

b = np.arange(8)

print(f"x:\n {x}")
print(f"Wf :\n {Wf}")
print(f"Wg :\n {Wg}")
print(f"Wi :\n {Wi}")
print(f"Wo :\n {Wo}")

stackW = np.hstack((Wf, Wg, Wi, Wo))
print(f"stackW: \n{stackW}")

X = np.dot(x, stackW)
print(f"X: \n {X}")

Xb = X + b
print(f"Xb: \n {Xb}")


print(f"f: \n {Xb[:, :H]}")
print(f"g: \n {Xb[:, H:H*2]}")
print(f"i: \n {Xb[:, H*2:H*3]}")
print(f"o: \n {Xb[:, H*3:H*4]}")