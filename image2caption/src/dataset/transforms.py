from torchvision.transforms import v2
import torch


def build_transforms():
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

    return train_transform, eval_transform