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
from src.models.model import Encoder
from torch.optim import Adam, SGD
import random

def main():

#######################
# yamlの読み込み
#######################
    DATASET_YAML = 'dataset.yaml'
    TRAIN_YAML = 'train.yaml'
    MODEL_YAML = 'model.yaml'

    dataset_config = load_yaml(f"./config/{DATASET_YAML}")

    DATA_DIR = dataset_config["DATA_DIR"]
    IMAGES_DIR = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["IMAGES"]}"
    CAPTIONS_CSV = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["CAPTIONS"]}"
    WORD_DICT_CSV = rf"{dataset_config['DATA_DIR']}/{dataset_config['PATHS']["WORD_DICT"]}"

    train_config = load_yaml(f"./config/{TRAIN_YAML}")
    TRAIN_RATIO = train_config["TRAIN_RATIO"]
    VALIDATE_RATIO = train_config["VALIDATE_RATIO"]
    TEST_RATIO = train_config["TEST_RATIO"]
    BATCH_SIZE = train_config["BATCH_SIZE"]
    SHOW_SCORE_INVTERVAL = train_config["SHOW_SCORE_INVTERVAL"]
    MAX_EPOCH = train_config["MAX_EPOCH"]
    LR = train_config["LR"]

    model_config = load_yaml(f"./config/{MODEL_YAML}")
    encoder_config = model_config["encoder"]

    encoder = Encoder(encoder_config)
    optim = SGD(
        encoder.parameters(),
        lr=LR
    )
    

#######################
# その他変数定義
#######################
    


#######################
# 学習・検証・テストデータの作成
#######################
    df = pl.read_csv(CAPTIONS_CSV)
    
    assert TRAIN_RATIO + VALIDATE_RATIO < 1.0
    
    unique_image_ids = df["image"].unique().to_list()
    random.Random(42).shuffle(unique_image_ids)

    n_images = len(unique_image_ids)
    n_train = int(n_images * TRAIN_RATIO)
    n_valid = int(n_images * VALIDATE_RATIO)

    # 画像名のセットを作成
    train_ids = set(unique_image_ids[:n_train])
    valid_ids = set(unique_image_ids[n_train:n_train + n_valid])
    test_ids = set(unique_image_ids[n_train + n_valid:])

    # 学習・検証・テストに使用する画像をフィルターしdataframeを作成
    train_df = df.filter(pl.col("image").is_in(train_ids))
    validate_df = df.filter(pl.col("image").is_in(valid_ids))
    test_df = df.filter(pl.col("image").is_in(test_ids))
    
    # データの重複を確認
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)

    # 学習用の画像加工トランスフォーマーを作成
    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        # v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 検証・テスト用の画像加工のトランスフォーマーを作成
    eval_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # データセット、データローダーを定義
    train_dataset = CustomDataset(train_df, IMAGES_DIR, transform=train_transform)
    validate_dataset = CustomDataset(validate_df, IMAGES_DIR, transform=eval_transform)
    test_dataset = CustomDataset(test_df, IMAGES_DIR, transform=eval_transform)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_config["BATCH_SIZE"],
        num_workers=4,
        shuffle=True)
    
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=train_config["BATCH_SIZE"],
        num_workers=4,
        shuffle=False
        )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_config["BATCH_SIZE"],
        num_workers=4,
        shuffle=False
        )

# #######################
# # 学習
# #######################

#     for i, (images, caption) in enumerate(train_dataloader):
#         Encoder.forward(images)
        
        

if __name__ == "__main__":
    main()