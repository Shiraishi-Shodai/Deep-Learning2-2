import kagglehub
import polars as pl
from matplotlib import pyplot as plt
import csv
import cv2
from torch.nn import functional as F
import torch
import glob
import cv2
from torchvision.transforms import v2
from torchvision.io import decode_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
from pathlib import Path
import japanize_matplotlib
import re
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.data import load_yaml, CustomDataset

print(sys.path)


def main():
    DATASET_YAML = 'dataset.yaml'
    TRAIN_YAML = 'train.yaml'

#######################
# yamlの読み込み
#######################

    dataset_config = load_yaml(f"./config/{DATASET_YAML}")
    DATA_DIR = dataset_config["DATA_DIR"]
    IMAGES_DIR = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["IMAGES"]}"
    CAPTIONS_CSV = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["CAPTIONS"]}"
    WORD_DICT_CSV = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["WORD_DICT"]}"

    train_config = load_yaml(f"./config/{TRAIN_YAML}")


#######################
# 学習・テストデータの作成
#######################
    df = pl.read_csv(CAPTIONS_CSV)

    train_df = df[:int(len(df) * (1 - train_config["TEST_SIZE"]))]
    test_df = df[int((len(df) * (1 - train_config["TEST_SIZE"]))):]

    transform = v2.Compose([
        v2.Resize((224, 224)),
        # v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(train_df, IMAGES_DIR, transform=transform)
    test_dataset = CustomDataset(test_df, IMAGES_DIR, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_config["BATCH_SIZE"], num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=train_config["BATCH_SIZE"])

#######################
# 学習
#######################
    for i, (image, caption) in enumerate(train_dataloader):
        print(f"{i+1}回目 {image.shape, len(caption)}")
        

if __name__ == "__main__":
    main()