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


