import torch
import yaml
import sys
from pathlib import Path
from data import load_yaml

sys.path.append(Path(__file__).parents[2])

config_file = rf'{Path(__file__).parents[2]}\config\train.yaml'
config = load_yaml(config_file)

def get_device(config):
    if config["DEVICE"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return torch.device(config["DEVICE"])