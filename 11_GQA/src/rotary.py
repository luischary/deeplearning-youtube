import torch
from typing import Tuple


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # insere dimensoes no meio mantedo a primeira nao batch e a ultima
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xq.shape = [B, L, H, D]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    q_len = xq.shape[-3]
    freqs_cis_ = reshape_for_broadcast(freqs_cis[:q_len, :], xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_cross_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Para ser usada quando q e k tem tamanhos diferentes
    xq.shape = [B, L, H, D]
    """
    q_len = xq.shape[-3]
    k_len = xk.shape[-3]

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis_q = reshape_for_broadcast(freqs_cis[:q_len, :], xq_)
    freqs_cis_k = reshape_for_broadcast(freqs_cis[:k_len, :], xk_)

    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
