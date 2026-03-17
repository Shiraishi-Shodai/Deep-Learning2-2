import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np

# =============
# データ準備
# =============

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# print(train_dataset)
# print(len(train_dataset))
# print(len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==================
# 2. CNNモデルの構築
# ==================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 畳み込み
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # プーリング
        self.pool = nn.MaxPool2d(2, 2)

        # 全結合層
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ==================
# 3. GPU設定
# ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.train()

# ==================
# 4. 損失関数の設定
# ==================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==================
# 5. 学習ループ
# ==================
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}]")


print("Training Finished!")


model.eval()
correct = 0
total = 0

batch_num = len(test_dataset) // 64 + 1 if len(test_dataset) % 64 != 0 else len(test_dataset) // 64
axes_size = 4


def show_result(axes_size, images, labels):
    cm = 1 / 2.54
    fig, ax = plt.subplots(axes_size, axes_size, figsize=(15*cm, 15*cm))
    
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    fig.set_facecolor("#eaeaf2")
    bg_ax.axis("off")

    for row in range(axes_size):
        for col in range(axes_size):
            image = images[row*axes_size + col]
            label = labels[row*axes_size + col]

            # ax[row, col].set_facecolor("none")
            ax[row, col].axis("off")
            ax[row, col].imshow(image.cpu().permute(1, 2, 0), cmap="gray")
            ax[row, col].text(
                0.5,
                -0.3,
                label,
                transform=ax[row, col].transAxes,
                ha="center"
            )


with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) != batch_num:
            continue

        if len(labels) < axes_size**2:
            continue

        show_images = images[:axes_size**2]
        show_predicted_labels = predicted[:axes_size**2].tolist()
        
        show_result(axes_size, show_images, show_predicted_labels)

print(f"Test Accuracy: {100 * correct / total}%")
plt.tight_layout()
# plt.style.use("ggplot") 
# plt.style.use("seaborn-v0_8")
plt.show()