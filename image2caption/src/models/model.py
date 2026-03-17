import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib
import cv2
import polars as pl
from pathlib import Path

class Encoder(nn.Module):
    def __init__(self, encoder_cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(**encoder_cfg["conv1"])
        self.pool1 = nn.MaxPool2d(**encoder_cfg["pool1"])
        self.conv2 = nn.Conv2d(**encoder_cfg["conv2"])
        self.pool2 = nn.MaxPool2d(**encoder_cfg["pool2"])
        self.conv3 = nn.Conv2d(**encoder_cfg["conv3"])
        self.pool3 = nn.MaxPool2d(**encoder_cfg["pool3"])

    def forward(self, xs):
        encode_out = self.conv1.forward(xs)
        encode_out = self.pool1.forward(encode_out)
        encode_out = self.conv2.forward(encode_out)
        encode_out = self.pool2.forward(encode_out)
        encode_out = self.conv3.forward(encode_out)
        encode_out = self.pool3.forward(encode_out)

        return encode_out
    
    def backward(self, encode_dout):
        encode_dout = self.pool3.backward(encode_dout)
        encode_dout = self.conv3.backward(encode_dout)
        encode_dout = self.conv2.backward(encode_dout)
        encode_dout = self.pool2.backward(encode_dout)
        encode_dout = self.pool1.backward(encode_dout)
        dxs = self.conv1.backward(encode_dout)

        return dxs


class Bridge(nn.Module):
    """CNNとLSTMの間を取り持つ
    """
    def __init__(self, encoder_cfg):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=linear_bias)
        self.relu = nn.ReLU()
    
    def forward(self, encode_out):
        bridge_out = self.linear.forward(encode_out)
        bridge_out = self.relu(bridge_out)
        
        return bridge_out
    
    def backward(self, bridege_dout):
        bridege_dout = self.relu.backward(bridege_dout)
        encode_dout = self.linear.backward(bridege_dout)

        return encode_dout
    

class Decoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_size, hidden_size, num_layers, lstm_bias, batch_first, dropout, in_features, out_features, linear_bias):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=lstm_bias, batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=linear_bias)
    
    def forward(self, bridge_out):
        decoder_out = self.embedding.forward(bridge_out)
        decoder_out = self.lstm.forward(decoder_out)
        decoder_out = self.linear.forward(decoder_out)

        return decoder_out
    
    def backward(self, dout):
        decoder_dout = self.linear.backward(dout)
        decoder_dout = self.lstm.backward(decoder_dout)
        bridge_dout = self.embedding.backward(decoder_dout)

        return bridge_dout
