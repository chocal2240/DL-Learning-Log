import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

class DeepMLP(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.Sigmoid
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), act(),
            nn.Linear(256, 256), act(), nn.Linear(256, 256), act(),
            nn.Linear(256, 256), act(), nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

def run_experiment(title, activation="relu", lr=0.001, max_batches=100):
    print("\n====", title, "====")
    model = DeepMLP(activation).to(device)
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for batch_idx, (X, y) in enumerate(train_loader):
        if batch_idx >= max_batches: break
        X, y = X.to(device), y.to(device)
        pred = model(X); loss = criterion(pred, y)
        optimizer.zero_grad(); loss.backward()
        grad_norms = [p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None and "weight" in n]
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"batch={batch_idx}, loss={loss.item():.4f}, grad_norms={[round(g, 6) for g in grad_norms]}")

run_experiment("正常训练：ReLU + lr=0.01", "relu", 0.01)
run_experiment("可能梯度爆炸：ReLU + lr=5.0", "relu", 5.0)
run_experiment("可能梯度消失：Sigmoid + lr=0.01", "sigmoid", 0.01)
