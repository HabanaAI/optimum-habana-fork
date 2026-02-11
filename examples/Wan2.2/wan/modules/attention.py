# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    USE_FSDPA = True
except ModuleNotFoundError:
    print(f"Cannot find module FusedSDPA")

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'hpu' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype).contiguous()
        k = k.transpose(1, 2).to(dtype).contiguous()
        v = v.transpose(1, 2).to(dtype).contiguous()

        if USE_FSDPA:
            out = FusedSDPA.apply(q, k, v, attn_mask, dropout_p, causal, None, "fast")
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


class FlashAttnV3Gaudi:
    def __init__ (self):
        self.q_chunk = int(os.environ.get("FA3_Q_CHUNK", 8192))
        self.kv_chunk = int(os.environ.get("FA3_KV_CHUNK", 8192))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
        fsdpa_mode: str = "fast",
        cp_size: int = 1,
        pad_len: int = 0,
        layout_head_first: bool = False,
        ) -> torch.Tensor:

        # Change to (batch, heads, seq_len, head_dim)
        if not layout_head_first:
            query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
        query_len = query.size(-2)
        key_len = key.size(-2)

        # In the case of cross-attn, use FusedSDPA.
        if  (query_len * cp_size) != key_len or (query_len <= 8192 and key_len <= 8192):
            output = FusedSDPA.apply(
                query,
                key,
                value,
                attention_mask,
                0.0,
                False,
                None,
                fsdpa_mode,
                None
            )
            return output.permute(0, 2, 1, 3).contiguous() if not layout_head_first else output

        #Flash Attention V3 for Full Attention
        linv_factor = 128.0 if fsdpa_mode == "fast" else 1.0

        if pad_len > 0:
            key = key[:, :, :-pad_len, :]
            value = value[:, :, :-pad_len, :]
            key_len = key.size(-2)

        num_query_chunk = int((query_len - 1) / self.q_chunk) + 1
        num_kv_chunk = int((key_len - 1) / self.kv_chunk) + 1

        final_hidden_list = []

        for query_idx in range(num_query_chunk):

            query_start = query_idx * self.q_chunk
            query_end = (query_idx + 1) * self.q_chunk if query_idx < num_query_chunk - 1 else query_len
            query_slice = query[..., query_start:query_end, :]

            out = None
            m = None
            linv = None

            for kv_idx in range(num_kv_chunk):

                kv_start = kv_idx * self.kv_chunk
                kv_end = (kv_idx + 1) * self.kv_chunk if kv_idx < num_kv_chunk - 1 else key_len

                key_slice = key[..., kv_start:kv_end, :]
                value_slice = value[..., kv_start:kv_end, :]

                block_out, block_m, block_linv, _ = torch.ops.hpu.sdpa_recomp_fwd(
                    query_slice,
                    key_slice,
                    value_slice,
                    None,
                    0.0,
                    1 / math.sqrt(query.shape[-1]),
                    False,
                    True,
                    fsdpa_mode,
                    None, #vsl,
                    "left",
                )

                if kv_idx == 0:
                    out = block_out.to(torch.float32)
                    m = block_m.to(torch.float32)
                    linv = block_linv.to(torch.float32) * linv_factor
                else:
                    block_linv = block_linv.to(torch.float32) * linv_factor
                    block_m = block_m.to(torch.float32)
                    block_out = block_out.to(torch.float32)
                    new_m = torch.maximum(m, block_m)
                    l_rescaled = (1.0 / linv) * torch.exp(m - new_m)
                    block_l_rescaled = (1.0 / block_linv) * torch.exp(block_m - new_m)
                    new_linv = 1.0 / (l_rescaled + block_l_rescaled)
                    out = (l_rescaled * new_linv) * out + (block_l_rescaled * new_linv) * block_out
                    linv = new_linv
                    m = new_m

            final_hidden_list.append(out.to(query.dtype))

        output = torch.cat(final_hidden_list, dim=-2)

        return output.permute(0, 2, 1, 3).contiguous() if not layout_head_first else output
