import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer import FeedForward, get_sinusoidal_embeddings
from src.attention import QKVMultiheadAttention, create_causal_mask
from src.tokenizer import SPECIAL_TOKENS


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


class EncoderForClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        last_hidden_states = self.encoder(x)
        pooled = last_hidden_states.mean(dim=1)  # Pooling simples: média dos tokens
        logits = self.classifier(pooled)
        return logits

    def train_step(self, batch):
        x, y = batch
        logits = self(x)
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())
        else:
            loss = F.cross_entropy(logits, y)
        return loss

    @torch.no_grad()
    def valid_step(self, batch):
        x, y = batch
        logits = self(x)
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(), y.float(), reduction="none"
            )
        else:
            loss = F.cross_entropy(logits, y, reduction="none")
        return loss


class EncoderMLM(nn.Module):
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

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )

        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        last_hidden_states = self.encoder(x)
        logits = self.mlm_head(last_hidden_states)

        return logits

    def train_step(self, batch):
        x, y = batch
        logits = self(x)

        y = y.reshape((-1,))
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), y, ignore_index=-100
        )
        return loss

    def valid_step(self, batch):
        x, y = batch
        logits = self(x)

        y = y.reshape((-1,))
        loss = F.cross_entropy(
            logits.reshape((-1, self.vocab_size)),
            y,
            reduction="none",
            ignore_index=-100,
        )
        return loss
