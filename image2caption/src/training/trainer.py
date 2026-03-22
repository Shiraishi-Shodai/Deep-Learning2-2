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
from utils.device import get_device


class Trainer:
    def __init__(self, model, train_loader, validate_loader, criterion, optimizer, config):
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = get_device(config)
        # モデルの重みをcudaで使用する
        self.model = model.to(self.device)
    
    def train(self):
        for epoch in range(self.ecpochs):
            self.train_one_epoch()
            self.validate()
    
    def train_one_epoch(self):
        self.model.train()

        for batch in self.dataloader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            loss = self.criterion(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def validate(self):
        self.model.eval()

        # 勾配計算を行わない
        with torch.no_grad():
            for batch in self.val_loader:
                pass
            
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def log(self, loss):
        print(loss)