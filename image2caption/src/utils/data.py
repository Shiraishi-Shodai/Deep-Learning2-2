import yaml
from torchvision.transforms import v2
from torchvision.io import decode_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import cv2


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data