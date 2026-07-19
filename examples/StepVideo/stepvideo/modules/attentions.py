import math
import os

import torch
import torch.nn as nn
from einops import rearrange

from stepvideo.distributed.parallel_state import (
    gather_forward,
    get_sequence_parallel_world_size,
)

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    USE_FSDPA = True
except ModuleNotFoundError:
    print("Cannot find module FusedSDPA")

try:
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except ImportError:
    xFuserLongContextAttention = None


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fav3 = FlashAttnV3Gaudi()
    
    def attn_processor(self, attn_type):
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        else:
            raise Exception('Not supported attention type...')

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:   ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        if USE_FSDPA:
            x = self.fav3.forward(q, k, v, attention_mask=attn_mask)
        else:
            q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
            )
            x = rearrange(x, 'b h s d -> b s h d')
        return x        

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        causal=False,
        **kwargs
    ):

        cp_size = get_sequence_parallel_world_size()
        k = gather_forward(k, dim=1)
        v = gather_forward(v, dim=1)

        x = self.fav3.forward(q, k, v, cp_size=cp_size)
        if cp_size > 1:
            torch.hpu.synchronize()

        return x


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

