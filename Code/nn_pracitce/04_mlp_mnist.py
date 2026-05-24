import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = TwoLayerMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
for epoch in range(5):
    model.train(); total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad(); loss.backward()
        grad_fc1 = model.fc1.weight.grad.norm().item()
        grad_fc2 = model.fc2.weight.grad.norm().item()
        optimizer.step(); total_loss += loss.item()
    avg_loss = total_loss / len(train_loader); losses.append(avg_loss)
    print(f"epoch={epoch+1}, loss={avg_loss:.4f}, grad_fc1={grad_fc1:.4f}, grad_fc2={grad_fc2:.4f}")

model.eval(); correct = total = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        correct += (pred.argmax(dim=1) == y).sum().item(); total += y.size(0)
print("test accuracy:", correct / total)
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("2-layer MLP on MNIST"); plt.show()
