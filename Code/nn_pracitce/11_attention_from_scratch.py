import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights

B, T, D = 2, 4, 8
Q = torch.randn(B, T, D); K = torch.randn(B, T, D); V = torch.randn(B, T, D)
output, weights = scaled_dot_product_attention(Q, K, V)
print("Q:", Q.shape); print("K:", K.shape); print("V:", V.shape)
print("weights:", weights.shape); print("output:", output.shape)
mask = torch.tril(torch.ones(T, T)).unsqueeze(0)
masked_output, masked_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
print("masked_weights shape:", masked_weights.shape)
