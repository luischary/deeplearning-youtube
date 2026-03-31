import torch.nn as nn
import torch


def create_causal_mask(seq_len):
    # Máscara triangular inferior (causal mask)
    mask = (
        torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    return mask


def QKVAttention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask=None,
    return_att_matrix=False,
):
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    att_matrix = torch.softmax(scores, dim=-1)
    att = att_matrix @ v
    if return_att_matrix:
        return att, att_matrix
    return att


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.Wo = nn.Linear(d_model, d_model)

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

    def break_heads(self, x):
        B, LEN, D = x.shape
        return x.reshape(B, LEN, self.n_heads, self.head_dim).transpose(1, 2)

    def concatenate_heads(self, x):
        B, N_HEADS, LEN, HEAD_DIM = x.shape
        return x.transpose(1, 2).contiguous().view(B, LEN, self.d_model)

    def forward(self, x, mask=None, return_att_matrix=False):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = self.break_heads(q)
        k = self.break_heads(k)
        v = self.break_heads(v)

        att, att_matrix = QKVAttention(q, k, v, mask, return_att_matrix=True)

        att = self.concatenate_heads(att)
        # em um modelo de verdade, Wo seria um parâmetro treinável, mas aqui só estamos ilustrando o processo
        att = self.Wo(att)

        if return_att_matrix:
            return att, att_matrix
        return att
