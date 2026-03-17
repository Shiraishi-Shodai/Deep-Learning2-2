import sys
from pathlib import Path

# プロジェクトルート (image2caption/) を取得
root = Path(__file__).resolve().parent.parent.parent
# プロジェクトルートとその親 (Deep-Learning2/) をパスに追加
sys.path.extend([str(root), str(root.parent)])

import polars as pl
from matplotlib import pyplot as plt
import csv
import cv2
from common.np import *
from torch.nn import functional as F
import torch
from sklearn.model_selection import train_test_split
import kagglehub


# =================================
# read csv data
# =================================
path = r"C:\Users\siran\.cache\kagglehub\datasets\adityajn105\flickr8k\versions\1"
data_file_path = Path(rf'{path}\captions.csv')
dict_path = Path(rf'{path}\word_dict.csv')
data = pl.read_csv(str(data_file_path))

word_dataframe = pl.read_csv(str(dict_path))
print(data.describe())
print(word_dataframe.describe())

# =================================
# DataLoaderの取得
# =================================
