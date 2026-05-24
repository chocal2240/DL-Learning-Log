import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx, embed_dim=128, hidden_size=128, num_classes=2):
        super().__init__(); self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True); self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        emb = self.embedding(x); output, (h_n, c_n) = self.rnn(emb); return self.fc(h_n[-1])

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx, embed_dim=128, hidden_size=128, num_classes=2):
        super().__init__(); self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True); self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        emb = self.embedding(x); output, h_n = self.rnn(emb); return self.fc(h_n[-1])

# 用法：在 09_lstm_imdb_colab.py 中把模型替换为这里的两个类。
# lstm_model = LSTMClassifier(len(vocab), pad_idx)
# gru_model = GRUClassifier(len(vocab), pad_idx)
# print(count_parameters(lstm_model))
# print(count_parameters(gru_model))
