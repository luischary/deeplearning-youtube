import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer import FeedForward, get_sinusoidal_embeddings
from src.attention import (
    CachedQKVMultiheadAttention,
    QKVMultiheadAttention,
    create_causal_mask,
)
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


class DecoderLM(nn.Module):
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

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.last_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embeds = self.decoder(x)
        embeds = self.last_norm(embeds)
        logits = self.proj(embeds)
        return logits

    def train_step(self, batch):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(
            logits.reshape((-1, self.vocab_size)),
            y.reshape((-1,)),
            ignore_index=SPECIAL_TOKENS["<PAD>"],
        )
        return loss

    @torch.no_grad()
    def valid_step(self, batch):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(
            logits.reshape((-1, self.vocab_size)),
            y.reshape((-1,)),
            ignore_index=SPECIAL_TOKENS["<PAD>"],
            reduction="none",
        )
        return loss


class CacheTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        hidden_size: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = CachedQKVMultiheadAttention(
            n_heads=n_heads, d_model=embed_dim, max_len=max_len
        )
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


class CachedDecoder(nn.Module):
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
                CacheTransformerBlock(
                    embed_dim=embed_dim,
                    n_heads=num_heads,
                    hidden_size=hidden_size,
                    max_len=max_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, position: int = 0):
        B, L = x.shape

        # embeddings
        x = self.embedding(x)
        # positional encoding
        pos = self.pos_embedding[position : position + L].to(x.device)

        x = x + pos

        mask = self.att_mask[:, :, :L, :L].repeat(B, 1, 1, 1).to(x.device)
        for block in self.blocks:
            x = block(x, att_mask=mask)

        return x

    def reset_cache(self):
        for block in self.blocks:
            block.attention.reset_cache()


class CachedDecoderLM(nn.Module):
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

        self.decoder = CachedDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.last_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, position: int = 0):
        embeds = self.decoder(x, position=position)
        embeds = self.last_norm(embeds)
        logits = self.proj(embeds)
        return logits

    @torch.no_grad()
    def generate(
        self, input_ids, eos_token_id, temperature: float = 1.0, top_p: float = 0.9
    ):
        self.decoder.reset_cache()

        generated = input_ids
        first_run = True  # para saber se manda tudo ou vai token a token
        pos = 0
        while generated.shape[1] < self.max_len:
            if first_run:
                logits = self(generated, position=pos)
                first_run = False
                pos += generated.shape[1]
            else:
                logits = self(generated[:, -1:], position=pos)
                pos += 1
            next_token_logits = logits[:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = -float("Inf")

            next_token_id = torch.multinomial(
                F.softmax(next_token_logits, dim=-1), num_samples=1
            )
            generated = torch.cat([generated, next_token_id], dim=-1)

            if next_token_id.item() == eos_token_id:
                break

        return generated.squeeze().tolist()
