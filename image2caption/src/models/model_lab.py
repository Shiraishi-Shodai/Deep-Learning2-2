import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib
import cv2
import polars as pl
from pathlib import Path

path = r"C:\Users\siran\.cache\kagglehub\datasets\adityajn105\flickr8k\versions\1"
csv_file_path = Path(r'C:\Users\siran\.cache\kagglehub\datasets\adityajn105\flickr8k\versions\1\captions.csv')
dataFrame = pl.read_csv(csv_file_path)
x = dataFrame.item(-1, 0)
x = Path(rf'{path}\Images\{x}')

image = cv2.imread(x)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(image)
image = image.float()
image = image.permute(2, 0, 1)
image = image.unsqueeze(0)

# 単純な線形モデル
# class Affine(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(in_dim, out_dim))
#         self.b = nn.Parameter(torch.zeros(out_dim))

#     def forward(self, x):
#         return x @ self.W + self.b


# x = torch.tensor([1, 0, 2], dtype=torch.float)
# model = Affine(3, 2)

# y = model.forward(x)
# print(y)

# 畳み込み層
# class Conv(nn.Module):
#     def __init__(self, in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

#     def forward(self, x):
#         return self.c1(x)

# # plt.imshow(image)
# # plt.show()

# model = Conv()

# y = model.forward(image)
# print(y.shape)

# y = y.squeeze()
# print(y.shape)

# y = y.permute(1, 2, 0)
# print(y.shape)

# y = y.detach().numpy().copy()
# print(y.shape, type(y))

# plt.imshow(y[:, :, :4])
# plt.show()


# プーリング層
# class Pool(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2, 2)
    
#     def forward(self, x):
#         x = self.pool(x)
#         return x

# model = Pool()
# y = model(image)
# print(y.shape)

# y = y.squeeze()
# print(y.shape)
# y = y.permute(1, 2, 0)
# print(y.shape)

# y = y.detach().numpy().copy()
# y = (y * 255).astype(np.uint8)
# print(np.max(y))
# print(np.min(y))
# print(np.max((y * 255).astype(np.uint8)))
# print(np.min((y * 255).astype(np.uint8)))
# plt.imshow(y)
# plt.show()

# CNN
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
#         # self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         # x = self.relu(x)
#         return x

# model = SimpleCNN()
# y = model(image)
# y = y.squeeze().permute(1, 2, 0).detach().numpy().copy()
# y = (y - y.min()) / (y.max() - y.min())

# plt.imshow(y[:, :, :4])
# plt.show()