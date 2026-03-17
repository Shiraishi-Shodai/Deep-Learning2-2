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


class CustomDataset(Dataset):
    def __init__(self, train_df, IMAGES_DIR, transform=None):
        self.transform = transform
        self.df = train_df
        self.IMAGES_DIR = IMAGES_DIR
    
    def __getitem__(self, idx):

        image_path, caption = self.df.item(idx, "image"), self.df.item(idx, "caption")
        image = self.__read_img(image_path)

        if self.transform:
            image = self.transform(image)
        
        return image, caption

    def __len__(self):
        return len(self.df)

    def __read_img(self, image_path):
        image = cv2.imread(rf"{self.IMAGES_DIR}\{image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image