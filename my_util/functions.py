import numpy as np

def MSE(X, y):
    N = X.shape[0]
    loss = (1/N) * (1/2) * np.sum((X - y)**2)
    return loss