# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_wan import WanAttention, _get_added_kv_projections, _get_qkv_projections
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from torch import nn

from ...distributed import parallel_state
from .embeddings import RotaryPosEmbedding
from .qwenimage_transformer import apply_rotary_emb_qwen_gaudi


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, invAttnHead=None):
        return torch.ops.hpu.softmax_fp8(x, dim, None, None, invAttnHead)


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)


# ScaledDotProductAttention is based on torch.nn.functional.scaled_dot_product_attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.bmm1 = Matmul()
        self.bmm2 = Matmul()
        self.softmax = Softmax()

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        # Efficient implementation:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        invAttnHead = torch.tensor(scale_factor, dtype=torch.float32).to("hpu")
        attn_bias = torch.zeros(L, S, dtype=query.dtype)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        if S < 128:
            attn_weight = self.bmm1(key, query.transpose(-2, -1))
            attn_weight = self.softmax(attn_weight, dim=-2, invAttnHead=invAttnHead)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return self.bmm2(attn_weight.transpose(-2, -1), value)
        else:
            attn_weight = self.bmm1(query, key.transpose(-2, -1))
            attn_weight = self.softmax(attn_weight, dim=-1, invAttnHead=invAttnHead)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return self.bmm2(attn_weight, value)


class FlashAttnV3Gaudi:
    def __init__(self):
        self.q_chunk = int(os.environ.get("FA3_Q_CHUNK", 8192))
        self.kv_chunk = int(os.environ.get("FA3_KV_CHUNK", 8192))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fsdpa_mode: str = "fast",
        cp_size: int = 1,
        pad_len: int = 0,
        ) -> torch.Tensor:

        # Change to (batch, heads, seq_len, head_dim)
        query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
        query_len = query.size(-2)
        key_len = key.size(-2)

        # In the case of cross-attn, use FusedSDPA.
        if (query_len * cp_size) != key_len or query_len <= 8192:
            output = FusedSDPA.apply(query, key, value, attention_mask, 0.0, False, None, fsdpa_mode, None)
            return output.permute(0, 2, 1, 3).contiguous()

        # Flash Attention V3 for Full Attention
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

        return output.permute(0, 2, 1, 3).contiguous()


# Copied from diffusers.models.attention_processor.AttnProcessor2_0
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attention_module=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.attention_module = attention_module

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        if os.environ.get("PATCH_SDPA") is not None:
            hidden_states = self.attention_module(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            import habana_frameworks.torch.hpu as ht
            from habana_frameworks.torch.hpex.kernels import FusedSDPA

            with ht.sdp_kernel(enable_recompute=True):
                hidden_states = FusedSDPA.apply(query, key, value, attention_mask, 0.0, False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


#  FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
    ):
        query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
        out = self._hpu_kernel_fsdpa.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            softmax_mode,
            recompute_mode,
            valid_sequence_lengths,
            padding_side,
        )
        return out.permute(0, 2, 1, 3)


class GaudiDistributedAttention(torch.nn.Module):
    def __init__(self, hpu_module_fsdpa: ModuleFusedSDPA):
        super().__init__()
        self._hpu_module_fsdpa = hpu_module_fsdpa
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
            from deepspeed.sequence.layer import DistributedAttention

            self._hpu_module_fsdpa_distributed = DistributedAttention(
                self._hpu_module_fsdpa, parallel_state.get_sequence_parallel_group(), 2, 1
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        dropout_p: float,
        is_casual,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
    ):
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
            return self._hpu_module_fsdpa_distributed(
                query,
                key,
                value,
                0,  # As the shape for inputs is [B, S, N, H]
                None,
                attn_mask,
                dropout_p,
                is_casual,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
            )
        else:
            return self._hpu_module_fsdpa(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_casual,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
            )


class CogVideoXAttnProcessorGaudi:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py#L1896
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = RotaryPosEmbedding.apply(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = RotaryPosEmbedding.apply(key[:, :, text_seq_length:], image_rotary_emb)

        softmax_mode = "None" if attn.training else "fast"
        hidden_states = self.fused_scaled_dot_product_attention(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            attention_mask,
            0.0,
            False,
            None,
            softmax_mode,
            False,
            None,
            "None",
        )

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class GaudiJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections.
    Copied from JointAttnProcessor2_0.forward: https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/models/attention_processor.py
        * Modified SDPA to use Gaudi fused SDPA kernel
        * Modified RMSNorm to use fast Gaudi fused RMSNorm kernel
    """

    def __init__(self, is_training=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.is_training = is_training

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # Self-attention: `sample` projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

        use_stages = False
        bwd_mode = 0
        fast_math = True

        if attn.norm_q is not None:
            query = FusedRMSNorm.apply(query, attn.norm_q.weight, attn.norm_q.eps, use_stages, bwd_mode, fast_math)
        if attn.norm_k is not None:
            key = FusedRMSNorm.apply(key, attn.norm_k.weight, attn.norm_k.eps, use_stages, bwd_mode, fast_math)

        # Cross-attention: `context` projections
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = FusedRMSNorm.apply(
                    encoder_hidden_states_query_proj,
                    attn.norm_added_q.weight,
                    attn.norm_added_q.eps,
                    use_stages,
                    bwd_mode,
                    fast_math,
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = FusedRMSNorm.apply(
                    encoder_hidden_states_key_proj,
                    attn.norm_added_k.weight,
                    attn.norm_added_k.eps,
                    use_stages,
                    bwd_mode,
                    fast_math,
                )

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # Fast FSDPA is not supported in training mode
        fsdpa_mode = "None" if self.is_training else "fast"
        hidden_states = FusedSDPA.apply(query, key, value, None, 0.0, False, None, fsdpa_mode, None)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def apply_rotary_emb_hpu(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py#L697
    """
    cos_, sin_ = freqs_cis  # [S, D]

    cos = cos_[None, None]
    sin = sin_[None, None]
    cos, sin = cos.to(xq.device), sin.to(xq.device)

    xq_out = torch.ops.hpu.rotary_pos_embedding(xq, sin, cos, None, 0, 1)
    xk_out = torch.ops.hpu.rotary_pos_embedding(xk, sin, cos, None, 0, 1)

    return xq_out, xk_out


class GaudiFluxAttnProcessor2_0:
    """
    Adapted from:
    https://github.com/huggingface/diffusers/blob/ed4efbd63d0f6b271894bc404b12f512d6b764e5/src/diffusers/models/attention_processor.py#L2275
      * Modified SDPA to use Gaudi fused SDPA kernel
      * Modified RoPE to use native PAIRWISE mode ordering HPU RoPE kernel
      * Modified RMSNorm to use fast Gaudi fused RMSNorm kernel
    """

    def __init__(self, is_training=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.is_training = is_training
        self.fav3 = FlashAttnV3Gaudi()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # Self-attention: `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Prepare QKV
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply RMSNorm to Q and K
        from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

        use_stages = False
        bwd_mode = 0
        fast_math = True

        if attn.norm_q is not None:
            query = FusedRMSNorm.apply(query, attn.norm_q.weight, attn.norm_q.eps, use_stages, bwd_mode, fast_math)
        if attn.norm_k is not None:
            key = FusedRMSNorm.apply(key, attn.norm_k.weight, attn.norm_k.eps, use_stages, bwd_mode, fast_math)

        # Attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # Cross-attention: `context` projections
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # Apply RMSNorm to Q and K context projections
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = FusedRMSNorm.apply(
                    encoder_hidden_states_query_proj,
                    attn.norm_added_q.weight,
                    attn.norm_added_q.eps,
                    use_stages,
                    bwd_mode,
                    fast_math,
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = FusedRMSNorm.apply(
                    encoder_hidden_states_key_proj,
                    attn.norm_added_k.weight,
                    attn.norm_added_k.eps,
                    use_stages,
                    bwd_mode,
                    fast_math,
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query, key = apply_rotary_emb_hpu(query, key, image_rotary_emb)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        # Fast FSDPA is not supported in training mode
        fsdpa_mode = "None" if self.is_training else "fast"
        hidden_states = self.fav3.forward(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attention_mask=attention_mask, fsdpa_mode=fsdpa_mode)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class GaudiWanAttnProcessor:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/transformers/transformer_wan.py#L67
    This class copied from `WanAttnProcessor` and overrides methods to use Gaudi-specific implementations.
    Add a func _native_attention which uses FusedSDPA on Gaudi
    Use hpex.kernels.apply_rotary_pos_emb on Gaudi
    """

    _attention_backend = None

    def __init__(self, is_training=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        self.is_training = is_training
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None
        self.fused_scaled_dot_product_attention_distributed = None
        self.use_sp = os.getenv("USE_SP", "True").lower() not in ("0", "false", "False")
        self.cp_size = parallel_state.get_sequence_parallel_world_size()
        self.fav3 = FlashAttnV3Gaudi()

        if not self.use_sp and parallel_state.sequence_parallel_is_initialized() and self.cp_size > 1:
            self.fused_scaled_dot_product_attention_distributed = (
                GaudiDistributedAttention(self.fused_scaled_dot_product_attention) if FusedSDPA else None
            )

    def _native_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        # Fast FSDPA is not supported in training mode
        fsdpa_mode = "None" if self.is_training else "fast"

        if self.fused_scaled_dot_product_attention_distributed:
            out = self.fused_scaled_dot_product_attention_distributed(
                query,
                key,
                value,
                attn_mask,
                0.0,
                False,
                None,
                fsdpa_mode,
                False,
                None,
                "None",
            )
        else:
            out = self.fused_scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                fsdpa_mode,
                False,
                None,
                "None",
            )
        return out

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pad_len: int = 0,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            """
            Wan's ROPE is pairwised, like this:
            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)
            """
            from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingMode, apply_rotary_pos_emb

            query = apply_rotary_pos_emb(query, *rotary_emb, None, 0, RotaryPosEmbeddingMode.PAIRWISE)
            key = apply_rotary_pos_emb(key, *rotary_emb, None, 0, RotaryPosEmbeddingMode.PAIRWISE)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = self._native_attention(query, key_img, value_img, None, 0.0, False, None)

            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Add traditional SP:
        if self.use_sp and self.cp_size > 1:
            bs, kv_seq, num_head, head_dim = key.shape
            key = key.reshape(bs, kv_seq, -1)
            value = value.reshape(bs, kv_seq, -1)
            full_key = torch.empty(bs, kv_seq * self.cp_size, num_head * head_dim, dtype=key.dtype, device=key.device)
            full_value = torch.empty(
                bs, kv_seq * self.cp_size, num_head * head_dim, dtype=value.dtype, device=value.device
            )
            gather1 = torch.distributed.all_gather_into_tensor(
                full_key,
                key,
                group=parallel_state.get_sequence_parallel_group(),
                async_op=True,
            )
            torch.distributed.all_gather_into_tensor(
                full_value,
                value,
                group=parallel_state.get_sequence_parallel_group(),
                async_op=False,
            )
            gather1.wait()
            key = full_key.reshape(bs, kv_seq * self.cp_size, num_head, head_dim)
            value = full_value.reshape(bs, kv_seq * self.cp_size, num_head, head_dim)

            if attention_mask is not None:
                logger.warning("Applying attention_mask in SP is not well supported, set it as None.")
                attention_mask = None

        hidden_states = self.fav3.forward(query, key, value, attention_mask, fsdpa_mode="fast",
                                          cp_size=self.cp_size, pad_len=pad_len)

        if self.use_sp and self.cp_size > 1:
            torch.hpu.synchronize()

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class GaudiQwenDoubleStreamAttnProcessor2_0:
    """
    Adapted from:
    https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_qwenimage.py#L261
        * Modified SDPA to use Gaudi fused SDPA kernel/FA3
        * apply_rotary_emb_qwen use_real=True
        * support cp
        * Padding for encoder_hidden_states
    """

    _attention_backend = None

    def __init__(self, is_training=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.is_training = is_training
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None
        self.cp_size = parallel_state.get_sequence_parallel_world_size()
        self.fav3 = FlashAttnV3Gaudi()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        encoder_hidden_states_pad_len: int = 0,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("GaudiQwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen_gaudi(img_query, img_freqs)
            img_key = apply_rotary_emb_qwen_gaudi(img_key, img_freqs)
            txt_query = apply_rotary_emb_qwen_gaudi(txt_query, txt_freqs)
            txt_key = apply_rotary_emb_qwen_gaudi(txt_key, txt_freqs)
        if self.cp_size > 1:
            bs, img_kv_seq, num_head, head_dim = img_key.shape
            img_key = img_key.reshape(bs, img_kv_seq, -1)
            img_value = img_value.reshape(bs, img_kv_seq, -1)

            img_full_key = torch.empty(
                bs, img_kv_seq * self.cp_size, num_head * head_dim, dtype=img_key.dtype, device=img_key.device
            )
            img_full_value = torch.empty(
                bs, img_kv_seq * self.cp_size, num_head * head_dim, dtype=img_value.dtype, device=img_value.device
            )
            gather2 = torch.distributed.all_gather_into_tensor(
                img_full_key,
                img_key,
                group=parallel_state.get_sequence_parallel_group(),
                async_op=True,
            )
            torch.distributed.all_gather_into_tensor(
                img_full_value,
                img_value,
                group=parallel_state.get_sequence_parallel_group(),
                async_op=False,
            )
            gather2.wait()
            img_key = img_full_key.reshape(bs, img_kv_seq * self.cp_size, num_head, head_dim)
            img_value = img_full_value.reshape(bs, img_kv_seq * self.cp_size, num_head, head_dim)

        if encoder_hidden_states_pad_len > 0:
            txt_key = txt_key[:, :-encoder_hidden_states_pad_len, :]
            txt_value = txt_value[:, :-encoder_hidden_states_pad_len, :]

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Fast FSDPA is not supported in training mode
        fsdpa_mode = "None" if self.is_training else "fast"

        if joint_key.shape[1] < 8192:
            joint_hidden_states = self.fused_scaled_dot_product_attention(
                joint_query,
                joint_key,
                joint_value,
                attention_mask,
                0.0,
                False,
                None,
                fsdpa_mode,
                False,
                None,
                "None",
            )
        else:
            joint_hidden_states = self.fav3.forward(
                joint_query, joint_key, joint_value, fsdpa_mode=fsdpa_mode, cp_size=self.cp_size
            )

        if self.cp_size > 1:
            torch.hpu.synchronize()

        joint_hidden_states = joint_hidden_states.contiguous()

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


AttentionProcessor = Union[AttnProcessor2_0,]
