"""
建议在 Google Colab 运行。可能需要：
!pip install torchtext portalocker
若 torchtext 版本不兼容，可改用 Hugging Face datasets。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter: yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(IMDB(split="train")), specials=["<unk>", "<pad>"], max_tokens=20000)
vocab.set_default_index(vocab["<unk>"]); pad_idx = vocab["<pad>"]

def text_pipeline(text): return vocab(tokenizer(text))[:300]
def label_pipeline(label): return 1 if label == "pos" else 0

def collate_batch(batch):
    texts, labels = [], []
    for label, text in batch:
        labels.append(label_pipeline(label)); texts.append(torch.tensor(text_pipeline(text), dtype=torch.long))
    return pad_sequence(texts, batch_first=True, padding_value=pad_idx), torch.tensor(labels, dtype=torch.long)

train_loader = DataLoader(list(IMDB(split="train"))[:8000], batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(IMDB(split="test"))[:2000], batch_size=64, shuffle=False, collate_fn=collate_batch)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_classes=2):
        super().__init__(); self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True); self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        emb = self.embedding(x); output, (h_n, c_n) = self.lstm(emb); return self.fc(h_n[-1])

model = LSTMClassifier(len(vocab)).to(device); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(3):
    model.train(); total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device); loss = criterion(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
    print(f"epoch={epoch+1}, loss={total_loss/len(train_loader):.4f}")
model.eval(); correct = total = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device); pred = model(X)
        correct += (pred.argmax(1) == y).sum().item(); total += y.size(0)
print("test accuracy:", correct / total)
