# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size

import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingMode, apply_rotary_pos_emb


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def rope_apply_gaudi(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    cos, sin = freqs
    cos = cos.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    sin = sin.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = x.to(torch.float64).reshape(s, n, -1, 2)
        x_real, x_imag = x_i.unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

        cos_i = torch.cat([
            cos[0][:f].reshape(f, 1, 1, -1).expand(f, h, w, -1),
            cos[1][:h].reshape(1, h, 1, -1).expand(f, h, w, -1),
            cos[2][:w].reshape(1, 1, w, -1).expand(f, h, w, -1)
        ],dim=-1).reshape(seq_len, 1, -1)

        sin_i = torch.cat([
            sin[0][:f].reshape(f, 1, 1, -1).expand(f, h, w, -1),
            sin[1][:h].reshape(1, h, 1, -1).expand(f, h, w, -1),
            sin[2][:w].reshape(1, 1, w, -1).expand(f, h, w, -1)
        ],dim=-1).reshape(seq_len, 1, -1)

        cos_i = cos_i.to(device=x.device)
        sin_i = sin_i.to(device=x.device)

        sp_size = get_world_size()
        sp_rank = get_rank()
        cos_i = pad_freqs(cos_i, s * sp_size)
        sin_i = pad_freqs(sin_i, s * sp_size)

        s_per_rank = s
        cos_i_rank = cos_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        sin_i_rank = sin_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]

        cos_i_rank = torch.repeat_interleave(cos_i_rank, 2, dim=2).reshape(s, 1, -1, 2)
        sin_i_rank = torch.repeat_interleave(sin_i_rank, 2, dim=2).reshape(s, 1, -1, 2)

        x_i = (x_i.float() * cos_i_rank + x_rotated.float() * sin_i_rank).flatten(2).to(x.dtype)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output)


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    freqs = self.rope(x[0])

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    pad_len = seq_len - seq_lens.max()
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.autocast(device_type="hpu", dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    cos, sin = freqs
    if pad_len > 0:
        cos = F.pad(cos, (0, 0, 0, 0, 0, pad_len))
        sin = F.pad(sin, (0, 0, 0, 0, 0, pad_len))
    cos = torch.chunk(cos, get_world_size(), dim=1)[get_rank()]
    sin = torch.chunk(sin, get_world_size(), dim=1)[get_rank()]
    freqs = (cos, sin)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)
        htcore.mark_step()

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, pad_len, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    q = apply_rotary_pos_emb(q, *freqs, None, 0, RotaryPosEmbeddingMode.PAIRWISE)
    k = apply_rotary_pos_emb(k, *freqs, None, 0, RotaryPosEmbeddingMode.PAIRWISE)

    # Gather K/V for sequence parallel
    k = gather_forward(k, dim=1)
    v = gather_forward(v, dim=1)

    cp_size = get_world_size()
    x = self.fav3.forward(half(q), half(k), half(v), cp_size=cp_size, pad_len=pad_len)

    if cp_size > 1:
        torch.hpu.synchronize()

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
