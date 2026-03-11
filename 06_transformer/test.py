import torch

from src.model import Encoder, Decoder

encoder = Encoder(
    max_len=10, embed_dim=16, vocab_size=100, n_layers=4, n_heads=4, hidden_size=64
)

x = torch.randint(0, 100, (2, 10))
y = encoder(x)
print(y.shape)

decoder = Decoder(
    max_len=10, embed_dim=16, vocab_size=100, n_layers=4, n_heads=4, hidden_size=64
)

x = torch.randint(0, 100, (2, 10))
y = decoder(x)
print(y.shape)
