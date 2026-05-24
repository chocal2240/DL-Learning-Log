import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), nn.AvgPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16*5*5, 120), nn.Sigmoid(), nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10)
        )
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if not self.training: print(layer.__class__.__name__, x.shape)
        return self.classifier(x)

model = LeNet().to(device); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.eval(); sample_X, _ = next(iter(train_loader))
with torch.no_grad(): _ = model(sample_X[:2].to(device))
for epoch in range(3):
    model.train(); total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X); loss = criterion(pred, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
    print(f"epoch={epoch+1}, loss={total_loss/len(train_loader):.4f}")
