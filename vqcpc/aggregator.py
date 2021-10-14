import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import lru_cache
from math import sqrt

from .utils import HelperModule

class AttentionBlock(HelperModule):
    def build(self,
            in_dim: int,
            out_dim: int,
            mlp_dim: int,
            nb_heads: int,
            max_len: int = 1024,
            dropout: float = 0.1,
        ):
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.attn = RelativeGlobalAttention(in_dim, out_dim, nb_heads, max_len=max_len, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_dim),
        )
        self.res = in_dim == out_dim

    def forward(self, x):
        x = self.norm1(self.attn(x) + (x if self.res else 0.0))
        x = self.norm2(self.mlp(x) + x)
        return x

# https://jaketae.github.io/study/relative-positional-encoding/
class RelativeGlobalAttention(HelperModule):
    def build(self,
            in_dim: int,
            out_dim: int,
            nb_heads: int,
            max_len: int = 1024,
            dropout: float = 0.1,
        ):
        assert not out_dim % nb_heads, "output dim must be divisible by number of heads!"
        self.nb_heads = nb_heads
        self.max_len = max_len
        self.head_dim = out_dim // nb_heads

        self.to_qkv = nn.Linear(in_dim, 3*out_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_parameter('Er', nn.Parameter(torch.randn(max_len, self.head_dim)))
        self.register_buffer('mask', torch.tril(torch.ones(1, 1, max_len, max_len)))

    def forward(self, x):
        N, L, _ = x.shape
        q, k, v = torch.chunk(self.to_qkv(x), 3, dim=-1)

        q = q.reshape(N, self.nb_heads, L, -1)
        k = k.reshape(N, self.nb_heads, -1, L)
        v = v.reshape(N, self.nb_heads, L, -1)

        start = self.max_len - L
        Er_t = self.Er[start:, :].transpose(0, 1)
        QEr = q @ Er_t

        Srel = self.skew(QEr)

        QK_t = q @ k

        mask = self.mask[:, :, :L, :L]
        attn = (QK_t + Srel) / sqrt(q.shape[-1])
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (F.softmax(attn, dim=-1) @ v).reshape(N, L, -1)

        return self.dropout(out)

    def skew(self, QEr):
        padded = F.pad(QEr, (1, 0))
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        Srel = reshaped[:, :, 1:, :]
        return Srel

# aggregate using lightweight self-attention layer
class AggregatorGPT(HelperModule):
    def build(self,
            embed_dim: int = 64,
            out_dim: int = 64,
            mlp_dim: int = 256,
            nb_layers: int = 2, 
            nb_heads: int = 8,
            seq_len: int = 128,
            dropout: float = 0.1,
        ):
        assert nb_layers > 0, "number of layers must be greater than 0!"
        blocks = [AttentionBlock(embed_dim, out_dim, mlp_dim, nb_heads, seq_len, dropout=dropout)]
        if nb_layers > 1:
            blocks.extend([AttentionBlock(out_dim, out_dim, mlp_dim, nb_heads, seq_len, dropout=dropout) for _ in range(nb_layers - 1)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

# standard aggregator as in van den Oord, 2020
class AggregatorGRU(HelperModule):
    def build(self,
            in_channels:        int = 64,
            hidden_channels:    int = 256,
            nb_layers:          int = 2,
        ):
        self.gru = nn.GRU(in_channels, hidden_channels, batch_first=True, num_layers=nb_layers)

    def forward(self, z):
        return self.gru(z)[0]
