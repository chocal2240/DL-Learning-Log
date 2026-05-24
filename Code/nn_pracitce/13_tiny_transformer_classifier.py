import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
num_samples, seq_len, vocab_size = 5000, 12, 20
threshold = seq_len * vocab_size * 0.5
X = torch.randint(0, vocab_size, (num_samples, seq_len))
y = (X.sum(dim=1) > threshold).long()
dataset = TensorDataset(X, y)
train_dataset, test_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, num_classes=2, max_len=100):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(x) + self.position_embedding(positions)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

model = TinyTransformerClassifier(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(8):
    model.train(); total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        loss = criterion(model(batch_x), batch_y)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device); pred = model(batch_x)
            correct += (pred.argmax(1) == batch_y).sum().item(); total += batch_y.size(0)
    print(f"epoch={epoch+1}, loss={total_loss/len(train_loader):.4f}, test_acc={correct/total:.4f}")

sample_x, _ = next(iter(test_loader)); sample_x = sample_x[:2].to(device)
print("input token shape:", sample_x.shape)
with torch.no_grad(): out = model(sample_x)
print("model output shape:", out.shape)
