import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__(); assert d_model % num_heads == 0
        self.d_model = d_model; self.num_heads = num_heads; self.d_head = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model); self.W_v = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model)
    def split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
    def combine_heads(self, x):
        B, H, T, d_head = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_head)
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q)); K = self.split_heads(self.W_k(K)); V = self.split_heads(self.W_v(V))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        out = self.combine_heads(out)
        return self.W_o(out), weights

B, T, D, H = 2, 5, 16, 4
x = torch.randn(B, T, D)
mha = MultiHeadAttention(D, H)
out, weights = mha(x, x, x)
print("input:", x.shape)
print("output:", out.shape)
print("attention weights:", weights.shape)
