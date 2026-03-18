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
        encode_out = self.conv1(xs)
        encode_out = self.pool1(encode_out)
        encode_out = self.conv2(encode_out)
        encode_out = self.pool2(encode_out)
        encode_out = self.conv3(encode_out)
        encode_out = self.pool3(encode_out)

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
    def __init__(self, bridge_cfg):
        super().__init__()
        self.linear = nn.Linear(**bridge_cfg)
        self.relu = nn.ReLU()
    
    def forward(self, encode_out):
        bridge_out = self.linear(encode_out)
        bridge_out = self.relu(bridge_out)
        
        return bridge_out
    
    def backward(self, bridege_dout):
        bridege_dout = self.relu.backward(bridege_dout)
        encode_dout = self.linear.backward(bridege_dout)

        return encode_dout
    

class Decoder(nn.Module):
    def __init__(self, decoder_cfg):
        super().__init__()
        self.embedding = nn.Embedding(**decoder_cfg["embedding"])
        self.lstm = nn.LSTM(**decoder_cfg["lstm"])
        self.linear = nn.Linear(**decoder_cfg["linear"])
    
    def forward(self, xs, h):
        decoder_out = self.embedding(bridge_out)
        decoder_out = self.lstm(decoder_out)
        decoder_out = self.linear(decoder_out)

        return decoder_out
    
    def backward(self, dout):
        decoder_dout = self.linear.backward(dout)
        decoder_dout = self.lstm.backward(decoder_dout)
        bridge_dout = self.embedding.backward(decoder_dout)

        return bridge_dout

    
    def generate(self, h, start_id, sample_size, h0):

        sampled = []
        sample_id = start_id

        c0 = torch.zeros_like(h0)

        for _ in range(sample_size):
            x = torch.tensor(sample_id).reshape(1, 1)
            decoder_out = self.embedding(sample_id, (h0, c0))
            decoder_out = self.lstm(decoder_out)
            score = self.linear(decoder_out)

            sample_id = int(torch.argmax(score.flatten()))

            sampled.append(sample_id)
        
        return sampled


class Seq2seq(nn.Module):
    def __init__(self, encoder_cfg, bridge_cfg, decoder_cfg):
        self.encoder = Encoder(encoder_cfg)
        self.bridge = Bridge(bridge_cfg)
        self.decoder = Decoder(decoder_cfg)

    def forward(self, xs):
        encoder_out = self.encoder.forward(xs)
        h = self.bridge.forward(encoder_out)
        out = self.decoder.forward(h)

        return out

    def backward(self, dout):
        decoder_dout = self.decoder.backward(dout)
        bridge_dout = self.bridge.backward(decoder_dout)
        dxs = self.encoder.backward(bridge_dout)

        return dxs