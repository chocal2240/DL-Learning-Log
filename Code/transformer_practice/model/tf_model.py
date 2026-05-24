"""A readable PyTorch implementation of the original Transformer architecture."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

import config


def clones(module: nn.Module, count: int) -> nn.ModuleList:
    """Create ``count`` deep copies of a module."""

    return nn.ModuleList(copy.deepcopy(module) for _ in range(count))


class Embeddings(nn.Module):
    """Token embedding scaled by ``sqrt(d_model)``."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the Transformer paper."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positional_encoding = self.pe[:, : x.size(1)].to(dtype=x.dtype)
        return self.dropout(x + positional_encoding)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention."""

    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)

    attention_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    return attention_weights @ value, attention_weights


class MultiHeadedAttention(nn.Module):
    """Multi-head attention with learned query, key, value, and output projections."""

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn: torch.Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = [
            linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """Layer normalization implemented explicitly for learning purposes."""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized + self.bias


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network applied independently to each sequence position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Residual connection followed by dropout around a pre-normalized sublayer."""

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """One Transformer encoder block."""

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda normalized: self.self_attn(normalized, normalized, normalized, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Stacked Transformer encoder."""

    def __init__(self, layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """One Transformer decoder block."""

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sublayer[0](x, lambda normalized: self.self_attn(normalized, normalized, normalized, tgt_mask))
        x = self.sublayer[1](x, lambda normalized: self.src_attn(normalized, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """Stacked Transformer decoder."""

    def __init__(self, layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """Project decoder hidden states into target vocabulary log-probabilities."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """Encoder-decoder Transformer."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int | None = None,
    d_model: int = config.d_model,
    d_ff: int = config.d_ff,
    h: int | None = None,
    dropout: float = config.dropout,
    *,
    num_layers: int | None = None,
    num_heads: int | None = None,
) -> Transformer:
    """Build a Transformer model and initialize its weights."""

    layer_count = num_layers if num_layers is not None else (N or config.n_layers)
    head_count = num_heads if num_heads is not None else (h or config.n_heads)

    attention_layer = MultiHeadedAttention(head_count, d_model, dropout)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    positional_encoding = PositionalEncoding(d_model, dropout)

    model = Transformer(
        Encoder(EncoderLayer(d_model, copy.deepcopy(attention_layer), copy.deepcopy(feed_forward), dropout), layer_count),
        Decoder(
            DecoderLayer(
                d_model,
                copy.deepcopy(attention_layer),
                copy.deepcopy(attention_layer),
                copy.deepcopy(feed_forward),
                dropout,
            ),
            layer_count,
        ),
        nn.Sequential(Embeddings(d_model, src_vocab), copy.deepcopy(positional_encoding)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), copy.deepcopy(positional_encoding)),
        Generator(d_model, tgt_vocab),
    )

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model.to(config.device)
