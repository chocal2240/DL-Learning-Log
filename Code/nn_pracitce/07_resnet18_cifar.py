import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out

class ResNet18Small(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__(); self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(64); self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, 1); self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2); self.layer4 = self._make_layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]; self.in_channels = out_channels
        for _ in range(1, blocks): layers.append(BasicBlock(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x); x = torch.flatten(x, 1); return self.fc(x)

model = ResNet18Small().to(device); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(2):
    model.train(); total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        loss = criterion(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
    print(f"epoch={epoch+1}, loss={total_loss/len(train_loader):.4f}")
model.eval(); correct = total = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device); pred = model(X)
        correct += (pred.argmax(1) == y).sum().item(); total += y.size(0)
print("test accuracy:", correct / total)
