import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
batch_size = 256
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_inputs = 28 * 28
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(logits):
    exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y] + 1e-9).mean()

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

lr = 0.1
epochs = 5
for epoch in range(epochs):
    total_loss, total_acc, count = 0, 0, 0
    for X, y in train_loader:
        X = X.reshape(-1, num_inputs)
        logits = X @ W + b
        y_hat = softmax(logits)
        loss = cross_entropy(y_hat, y)
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad
            W.grad.zero_()
            b.grad.zero_()
        total_loss += loss.item()
        total_acc += accuracy(y_hat, y)
        count += 1
    print(f"epoch={epoch+1}, loss={total_loss/count:.4f}, train_acc={total_acc/count:.4f}")

test_acc, count = 0, 0
with torch.no_grad():
    for X, y in test_loader:
        X = X.reshape(-1, num_inputs)
        y_hat = softmax(X @ W + b)
        test_acc += accuracy(y_hat, y)
        count += 1
print("test_acc:", test_acc / count)
