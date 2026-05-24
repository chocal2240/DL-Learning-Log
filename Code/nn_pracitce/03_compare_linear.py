import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.linear(x)

model = LinearClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print("weight shape:", model.linear.weight.shape)
print("bias shape:", model.linear.bias.shape)

def evaluate():
    model.eval(); correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total

for epoch in range(5):
    model.train(); total_loss = 0
    for X, y in train_loader:
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch={epoch+1}, loss={total_loss/len(train_loader):.4f}, test_acc={evaluate():.4f}")
