
import sys
sys.path.append('.')
from common.np import *
import numpy as _np

print(f"NumPy version: {_np.__version__}")
try:
    if GPU:
        import cupy
        print(f"CuPy version: {cupy.__version__}")
    else:
        print("Running in CPU mode (NumPy only)")
except ImportError:
    print("CuPy not available")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

try:
    print("Creating array...")
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='f')
    print("x dtype:", x.dtype)
    print("Testing sigmoid...")
    y = sigmoid(x)
    print("Sigmoid result:", y)
    print("Done.")
except Exception as e:
    import traceback
    traceback.print_exc()
