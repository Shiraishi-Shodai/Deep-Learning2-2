import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# X = np.linspace(1, 30, 1000).reshape(-1, 1)
# y = np.sin(X).reshape(-1, 1)

# data = np.concatenate([X, y], axis=1)
# np.savetxt("../data/sin_data.csv", data, delimiter=",", fmt="%.2f", header="X,y", comments="")

df = pd.read_csv("../data/sin_data.csv")
X = df["X"].to_numpy()
y = df["y"].to_numpy()

plt.plot(X, y)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("original.png")