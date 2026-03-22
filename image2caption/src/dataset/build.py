from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import cv2

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

def create_dataloaders(train_df, validate_df, test_df, images_dir, batch_size, train_transform, eval_transform):
        # データセット、データローダーを定義
    # 検証・テスト用のデータはより本番データに近づけるためにフリップ等の前処理は使用しない。
    train_dataset = CustomDataset(train_df, images_dir, transform=train_transform)
    validate_dataset = CustomDataset(validate_df, images_dir, transform=eval_transform)
    test_dataset = CustomDataset(test_df, images_dir, transform=eval_transform)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        num_workers=4,
        shuffle=True # モデルがデータの並びに引っ張られにくくなるようにマイエポックデータをシャッフルする
        )
    
    #検証用
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
        )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
        )
    return train_dataloader, validate_dataloader, test_dataloader