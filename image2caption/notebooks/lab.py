import sys
from pathlib import Path

# プロジェクトルート (image2caption/) を取得
root = Path(__file__).resolve().parent.parent
# プロジェクトルートとその親 (Deep-Learning2/) をパスに追加
sys.path.extend([str(root), str(root.parent)])

import kagglehub
import polars as pl
from matplotlib import pyplot as plt
import csv
import cv2
# from common.np import *
from torch.nn import functional as F
import torch

# a = 'Hello World'
# b = 'bbb'
# a = b*2 + a

# print(a)

# df = pl.DataFrame(
#     {
#         "id": [1, 2, 3],
#         "word": ["a", "b", "c"],
#     }
# )


# mask = np.array([True, False, True])
# y = np.arange(1, 4)
# g = mask * y
# print(y)
# print(g)

# a = np.arange(12).reshape(3, -1)
# b = np.arange(3)
# ts = np.array([0, 2, 3])

# print(a, end="\n")
# print(b, end="\n")
# print(ts, end="\n")

# print(a[b, ts])

# mask = np.array([True, False])

# print(mask[:, np.newaxis])

# =================================
# 1. テキストファイルをcsvファイルに変換
# =================================

# text_file_path = Path(rf'{path}\captions.txt')
# csv_file_path = Path(rf'{path}\captions.csv')

def txt2csv(text_file_path, csv_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(csv_file_path, 'w', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)

        for line in lines:
            row = line.strip().split(',', 1)
            writer.writerow(row)

# =================================
# ワンホットベクトル実験
# =================================
# word_size = 5
# a = torch.arange(0, word_size)
# one_hot_vec = F.one_hot(a)
# print(a)
# print(one_hot_vec)


# =================================
# 画像を一枚表示
# =================================
# fisrt_img_name = data[0, 0]
# first_img_src = Path(rf'{path}\Images\{fisrt_img_name}')
# fisrt_captions = data.filter(pl.col('image') == fisrt_img_name)["caption"]

# # print(fisrt_img_name)
# print(fisrt_captions, type(fisrt_captions))

# fig = plt.figure()
# ax = plt.subplot(1, 1, 1)
# img = cv2.imread(first_img_src)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 各キャプションを表示する位置を取得
# caption_size = len(fisrt_captions)
# base_position_bottom = -0.025
# # start_position_bottom = -0.1
# caption_position_bottom = np.arange(base_position_bottom, base_position_bottom*caption_size, base_position_bottom)

# print(caption_position_bottom)
# print(type(caption_position_bottom))

# for caption, position_bottom in zip(fisrt_captions, caption_position_bottom):
#     ax.text(
#         0.5,
#         position_bottom,
#         caption,
#         transform=ax.transAxes,
#         ha="center"
#     )

# ax.imshow(img)
# ax.set_axis_off()
# plt.show()


# a = 40455
# print(a * 0.1)
# print(a * 0.8)