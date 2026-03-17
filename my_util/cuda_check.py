import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())
