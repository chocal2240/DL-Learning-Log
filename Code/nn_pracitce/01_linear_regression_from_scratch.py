import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

n = 200
X = torch.randn(n, 1)
true_w = torch.tensor([[3.5]])
true_b = torch.tensor([-2.0])
y = X @ true_w + true_b + 0.3 * torch.randn(n, 1)

w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
epochs = 100
losses = []

for epoch in range(epochs):
    y_pred = X @ w + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"epoch={epoch}, loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Linear Regression from Scratch")
plt.show()

print("true_w:", true_w.item(), "learned_w:", w.item())
print("true_b:", true_b.item(), "learned_b:", b.item())
