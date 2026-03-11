import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer import FeedForward, get_sinusoidal_embeddings
from src.attention import QKVMultiheadAttention, create_causal_mask


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, n_heads: int, hidden_size: int, dropout: float = 0.1
    ):
        super().__init__()

        self.attention = QKVMultiheadAttention(n_heads=n_heads, d_model=embed_dim)
        self.ff = FeedForward(
            input_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, att_mask=None):
        x_att = self.attention(x, mask=att_mask)
        x = self.norm_1(x + x_att)

        x_ff = self.ff(x)
        x = self.norm_2(x + x_ff)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = get_sinusoidal_embeddings(n_seq=max_len, d_model=embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.last_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        B, L = x.shape

        # embeddings
        x = self.embedding(x)
        # positional encoding
        pos = self.pos_embedding[:L].to(x.device)

        x = x + pos

        for block in self.blocks:
            x = block(x)

        x = self.last_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = get_sinusoidal_embeddings(n_seq=max_len, d_model=embed_dim)
        self.att_mask = create_causal_mask(max_len)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        B, L = x.shape

        # embeddings
        x = self.embedding(x)
        # positional encoding
        pos = self.pos_embedding[:L].to(x.device)

        x = x + pos

        mask = self.att_mask[:, :, :L, :L].repeat(B, 1, 1, 1).to(x.device)
        for block in self.blocks:
            x = block(x, att_mask=mask)

        return x
