import torch
import torch.nn as nn
import torch.nn.functional as F


class Modelo(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, dropout=0.0):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, h=None, c=None):
        B, L = tokens.shape

        x = self.embed(tokens)

        if h is None:
            h = torch.zeros((self.n_layers, B, self.hidden_size), device=x.device)
            c = torch.zeros((self.n_layers, B, self.hidden_size), device=x.device)

        x, (h, c) = self.rnn(x, (h, c))
        logits = self.proj(x)
        return logits, (h, c)

    def train_step(self, batch):
        tokens_x, tokens_y = batch

        logits, _ = self.forward(tokens_x)  # shape logits = [B, L, H]
        loss = F.cross_entropy(
            logits.reshape((-1, self.vocab_size)),
            tokens_y.reshape((-1,)),
            ignore_index=0,
        )
        return loss

    def valid_step(self, batch):
        tokens_x, tokens_y = batch

        logits, _ = self.forward(tokens_x)  # shape logits = [B, L, H]
        loss = F.cross_entropy(
            logits.reshape((-1, self.vocab_size)),
            tokens_y.reshape((-1,)),
            ignore_index=0,
            reduction="none",
        )
        return loss
