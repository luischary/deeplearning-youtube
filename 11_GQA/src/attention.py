import math

import torch.nn as nn
import torch.nn.functional as F
import torch

from src.transformer import KVCache
from src.rotary import apply_rotary_emb


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
        x = x.reshape(B, LEN, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, N_HEADS, LEN, HEAD_DIM]
        return x

    def concatenate_heads(self, x):
        B, N_HEADS, LEN, HEAD_DIM = x.shape
        return x.transpose(1, 2).contiguous().view(B, LEN, self.d_model)

    def forward(self, x, mask=None, rotary_freqs=None):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = self.break_heads(q)
        k = self.break_heads(k)
        v = self.break_heads(v)

        if rotary_freqs is not None:
            # precisa transpor para aplicar a função de rotary, que espera [B, N_HEADS, HEAD_DIM, LEN]
            q, k = apply_rotary_emb(q.transpose(2, 1), k.transpose(2, 1), rotary_freqs)
            q = q.transpose(2, 1)
            k = k.transpose(2, 1)

        att, att_matrix = QKVAttention(q, k, v, mask, return_att_matrix=True)

        att = self.concatenate_heads(att)
        # em um modelo de verdade, Wo seria um parâmetro treinável, mas aqui só estamos ilustrando o processo
        att = self.Wo(att)

        return att


class CachedQKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, max_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.Wo = nn.Linear(d_model, d_model)

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

        self.cache = KVCache(max_len=max_len)

    def break_heads(self, x):
        B, LEN, D = x.shape
        return x.reshape(B, LEN, self.n_heads, self.head_dim).transpose(1, 2)

    def concatenate_heads(self, x):
        B, N_HEADS, LEN, HEAD_DIM = x.shape
        return x.transpose(1, 2).contiguous().view(B, LEN, self.d_model)

    def forward(self, x, mask=None, rotary_freqs=None):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = self.break_heads(q)
        k = self.break_heads(k)
        v = self.break_heads(v)

        if rotary_freqs is not None:
            # precisa transpor para aplicar a função de rotary, que espera [B, N_HEADS, HEAD_DIM, LEN]
            q, k = apply_rotary_emb(q.transpose(2, 1), k.transpose(2, 1), rotary_freqs)
            q = q.transpose(2, 1)
            k = k.transpose(2, 1)

        # adiciona no cache e ja atualiza k e v para conterem toda a sequência até o momento
        k, v = self.cache.add(k, v)

        att = QKVAttention(q, k, v, mask, return_att_matrix=False)

        att = self.concatenate_heads(att)
        # em um modelo de verdade, Wo seria um parâmetro treinável, mas aqui só estamos ilustrando o processo
        att = self.Wo(att)

        return att

    def reset_cache(self):
        self.cache.reset()


class GQMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, group_size: int, d_model: int):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_heads % group_size == 0

        self.n_heads = n_heads
        self.n_groups = n_heads // group_size

        self.head_dim = d_model // n_heads
        self.Wo = nn.Linear(d_model, d_model)

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, int(self.n_groups * self.head_dim))
        self.proj_v = nn.Linear(d_model, int(self.n_groups * self.head_dim))

    def break_heads_v(self, x):
        B, LEN, D = x.shape
        x = x.view(B, LEN, -1, self.head_dim).transpose(
            1, 2
        )  # [B, N_HEADS, LEN, HEAD_DIM]
        return x

    def break_heads(self, x):
        B, LEN, D = x.shape
        x = x.view(B, LEN, -1, self.head_dim)  # [B, LEN, N_GROUPS, HEAD_DIM]
        return x

    def concatenate_heads(self, x):
        B, N_HEADS, LEN, HEAD_DIM = x.shape
        x = x.transpose(1, 2).contiguous().view(B, LEN, -1)
        return x

    def forward(self, x, mask=None, rotary_freqs=None):
        q = self.proj_q(x)  # [B, LEN, D_MODEL]
        k = self.proj_k(x)  # [B, LEN, N_GROUPS * HEAD_DIM]
        v = self.proj_v(x)  # [B, LEN, N_GROUPS * HEAD_DIM]

        q = self.break_heads(q)  # [B, N_HEADS, LEN, HEAD_DIM]
        k = self.break_heads(k)  # [B, LEN, N_GROUPS, HEAD_DIM]
        v = self.break_heads_v(v)  # [B, N_GROUPS, LEN, HEAD_DIM]

        if rotary_freqs is not None:
            q, k = apply_rotary_emb(
                q, k, rotary_freqs
            )  # [B, LEN, HEADS/GROUPS, HEAD_DIM]
            q = q.transpose(
                2, 1
            )  # deixa pronto para a atencao [B, HEADS/GROUPS, LEN, HEAD_DIM]
            k = k.transpose(2, 1)

        # precisa repetir k e v para chegar no tamanho de q
        k = k.repeat_interleave(self.n_heads // self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_groups, dim=1)

        att = QKVAttention(q, k, v, mask, return_att_matrix=False)

        att = self.concatenate_heads(att)
        att = self.Wo(att)

        return att


# def grouped_query_attention(q, k, v, mask=None):
#     """
#     q: [B, n_heads, seq_len_q, head_dim]
#     k: [B, n_groups, seq_len_k, head_dim]
#     v: [B, n_groups, seq_len_k, head_dim]
#     """

#     B, n_heads, seq_len_q, head_dim = q.shape
#     _, n_groups, seq_len_k, _ = k.shape

#     group_size = n_heads // n_groups

#     # [B, n_groups, group_size, seq_len_q, head_dim]
#     q = q.view(B, n_groups, group_size, seq_len_q, head_dim)

#     # attention scores:
#     # q: [B, G, S, seq_len_q, head_dim]
#     # k: [B, G, seq_len_k, head_dim]
#     # result: [B, G, S, seq_len_q, seq_len_k]
#     scores = torch.einsum("bgsqd,bgkd->bgsqk", q, k)
#     scores = scores / math.sqrt(head_dim)

#     if mask is not None:
#         # ajuste dependendo do formato da sua mask
#         scores = scores.masked_fill(mask == 0, float("-inf"))

#     att = F.softmax(scores, dim=-1)

#     # att: [B, G, S, seq_len_q, seq_len_k]
#     # v:   [B, G, seq_len_k, head_dim]
#     # out: [B, G, S, seq_len_q, head_dim]
#     out = torch.einsum("bgsqk,bgkd->bgsqd", att, v)

#     # volta para [B, n_heads, seq_len_q, head_dim]
#     out = out.reshape(B, n_heads, seq_len_q, head_dim)

#     return out


def grouped_query_attention(q, k, v, mask=None):
    """
    q: [B, n_heads, seq_len_q, head_dim]
    k: [B, n_groups, seq_len_k, head_dim]
    v: [B, n_groups, seq_len_k, head_dim]
    """
    B, n_heads, seq_len_q, head_dim = q.shape
    _, n_groups, seq_len_k, _ = k.shape
    group_size = n_heads // n_groups

    # 1. Reshape Q para isolar os grupos
    # [B, n_groups, group_size, seq_len_q, head_dim]
    q = q.view(B, n_groups, group_size, seq_len_q, head_dim)

    # 2. Reshape K e V para adicionar uma dimensão de 'singleton'
    # Isso permite o broadcasting com o group_size de Q
    # [B, n_groups, 1, seq_len_k, head_dim]
    k = k.unsqueeze(2)
    v = v.unsqueeze(2)

    # 3. Cálculo do Score (Matmul + Transpose)
    # k.transpose(-2, -1) vira [B, n_groups, 1, head_dim, seq_len_k]
    # O broadcasting acontece na dimensão 2 (1 -> group_size)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    if mask is not None:
        # A máscara precisa ser compatível com as 5 dimensões
        scores = scores.masked_fill(mask == 0, float("-inf"))

    att = F.softmax(scores, dim=-1)

    # 4. Out: [B, n_groups, group_size, seq_len_q, head_dim]
    out = torch.matmul(att, v)

    # 5. Volta para o formato original de Multi-Head
    # [B, n_heads, seq_len_q, head_dim]
    out = out.view(B, n_heads, seq_len_q, head_dim)

    return out


class CachedGQMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, group_size: int, d_model: int, max_len: int):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_heads % group_size == 0

        self.n_heads = n_heads
        self.n_groups = n_heads // group_size

        self.head_dim = d_model // n_heads
        self.Wo = nn.Linear(d_model, d_model)

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, int(self.n_groups * self.head_dim))
        self.proj_v = nn.Linear(d_model, int(self.n_groups * self.head_dim))

        self.cache = KVCache(max_len=max_len)

    def break_heads_v(self, x):
        B, LEN, D = x.shape
        x = x.view(B, LEN, -1, self.head_dim).transpose(
            1, 2
        )  # [B, N_HEADS, LEN, HEAD_DIM]
        return x

    def break_heads(self, x):
        B, LEN, D = x.shape
        x = x.view(B, LEN, -1, self.head_dim)  # [B, LEN, N_GROUPS, HEAD_DIM]
        return x

    def concatenate_heads(self, x):
        B, N_HEADS, LEN, HEAD_DIM = x.shape
        x = x.transpose(1, 2).contiguous().view(B, LEN, -1)
        return x

    def forward(self, x, mask=None, rotary_freqs=None):
        q = self.proj_q(x)  # [B, LEN, D_MODEL]
        k = self.proj_k(x)  # [B, LEN, N_GROUPS * HEAD_DIM]
        v = self.proj_v(x)  # [B, LEN, N_GROUPS * HEAD_DIM]

        q = self.break_heads(q)  # [B, N_HEADS, LEN, HEAD_DIM]
        k = self.break_heads(k)  # [B, LEN, N_GROUPS, HEAD_DIM]
        v = self.break_heads_v(v)  # [B, N_GROUPS, LEN, HEAD_DIM]

        if rotary_freqs is not None:
            q, k = apply_rotary_emb(
                q, k, rotary_freqs
            )  # [B, LEN, HEADS/GROUPS, HEAD_DIM]
            q = q.transpose(
                2, 1
            )  # deixa pronto para a atencao [B, HEADS/GROUPS, LEN, HEAD_DIM]
            k = k.transpose(2, 1)

        # cache antes do repeat para garantir que o cache só armazene uma versão de k e v, economizando memória
        k, v = self.cache.add(k, v)

        # precisa repetir k e v para chegar no tamanho de q
        # k = k.repeat_interleave(self.n_heads // self.n_groups, dim=1)
        # v = v.repeat_interleave(self.n_heads // self.n_groups, dim=1)

        # att = QKVAttention(q, k, v, mask, return_att_matrix=False)
        att = grouped_query_attention(q, k, v, mask)

        att = self.concatenate_heads(att)
        att = self.Wo(att)

        return att

    def reset_cache(self):
        self.cache.reset()
