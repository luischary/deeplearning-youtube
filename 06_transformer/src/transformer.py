import torch
import torch.nn as nn
import math


def get_sinusoidal_embeddings(n_seq, d_model):
    """
    Cria uma matriz de Positional Encoding senoidal.

    Args:
        n_seq: Comprimento da sequência (número de tokens).
        d_model: Dimensão do modelo (tamanho do embedding).
    """
    # 1. Inicializa a matriz de posições (n_seq, d_model)
    pe = torch.zeros((n_seq, d_model))

    # 2. Calcula as posições (0, 1, 2, ..., n_seq-1)
    pos = torch.arange(n_seq).reshape(-1, 1)

    # 3. Calcula o denominador (frequências)
    # Usamos exp e log para calculo vetorial e estabilidade numérica: 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    # 4. Aplica seno nas colunas pares e cosseno nas ímpares
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)

    return pe


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout: float = 0.1):
        super().__init__()

        self.f1 = nn.Linear(input_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.f2 = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        x = self.f1(x)
        x = nn.functional.relu(x)  # gelu e uma boa opcao
        x = self.dropout(x)
        x = self.f2(x)
        return x
