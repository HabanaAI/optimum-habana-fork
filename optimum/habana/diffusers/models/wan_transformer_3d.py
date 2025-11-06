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

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput

# from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from ...distributed import parallel_state


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def WanTransformer3DModleForwardGaudi(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py#L597
    add mark_step.
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    pad_len = 0
    attention_mask = None
    if parallel_state.sequence_parallel_is_initialized():
        bs, seq_len, _ = hidden_states.shape
        cp_size = parallel_state.get_sequence_parallel_world_size()
        # We need to ensure seq_len can be divided by cp_size
        if seq_len % cp_size != 0:
            padded_seq_len = (seq_len // cp_size + 1) * cp_size
            pad_len = padded_seq_len - seq_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            cos = F.pad(rotary_emb[0], (0, 0, 0, 0, 0, pad_len))
            sin = F.pad(rotary_emb[1], (0, 0, 0, 0, 0, pad_len))
            rotary_emb = (cos, sin)
            if timestep.ndim == 2:
                timestep = F.pad(timestep, (0, pad_len))

            use_mask = os.getenv("CP_USE_MASK", "False")
            use_mask = use_mask.lower() in ("1", "true", "True")
            if use_mask:
                attention_mask = torch.ones(bs, 1, seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
                attention_mask = F.pad(attention_mask, (0, pad_len, 0, pad_len)).bool()

            seq_len = padded_seq_len

        sp_seq_len = seq_len // parallel_state.get_sequence_parallel_world_size()
        start = sp_seq_len * parallel_state.get_sequence_parallel_rank()
        end = sp_seq_len * (parallel_state.get_sequence_parallel_rank() + 1)

        hidden_states = hidden_states[:, start:end, :]

        # timestep with 2 dims means expand_timesteps in config is True, and
        # We only need to split the timestep when it has 2 dim.
        expanded_timestep = False
        if timestep.ndim == 2:
            expanded_timestep = True
            timestep = timestep[:, start:end]

        cos = rotary_emb[0][:, start:end, :, :]
        sin = rotary_emb[1][:, start:end, :, :]
        rotary_emb = (cos, sin)

    # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()  # batch_size * seq_len
    else:
        ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    if ts_seq_len is not None:
        # batch_size, seq_len, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # 4. Transformer blocks
    import habana_frameworks.torch.core as htcore

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_mask
            )
            htcore.mark_step()
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_mask)
            htcore.mark_step()

    # 5. Output norm, projection & unpatchify
    if temb.ndim == 3:
        # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        # batch_size, inner_dim
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    if parallel_state.sequence_parallel_is_initialized():
        cp_size = parallel_state.get_sequence_parallel_world_size()
        bs, seq, dim = hidden_states.shape

        gather_hidden = torch.empty(bs, seq * cp_size, dim, dtype=hidden_states.dtype, device=hidden_states.device)
        gather1 = torch.distributed.all_gather_into_tensor(
            gather_hidden,
            hidden_states,
            group=parallel_state.get_sequence_parallel_group(),
            async_op=True,
        )

        gather_shift = None
        gather_scale = None
        if expanded_timestep:
            gather_shift = torch.empty(bs, seq * cp_size, dim, dtype=shift.dtype, device=shift.device)
            torch.distributed.all_gather_into_tensor(
                gather_shift,
                shift,
                group=parallel_state.get_sequence_parallel_group(),
                async_op=False,
            )

            gather_scale = torch.empty(bs, seq * cp_size, dim, dtype=scale.dtype, device=scale.device)
            torch.distributed.all_gather_into_tensor(
                gather_scale,
                scale,
                group=parallel_state.get_sequence_parallel_group(),
                async_op=False,
            )

        gather1.wait()

        hidden_states = gather_hidden.reshape(bs, seq * cp_size, dim)
        shift = gather_shift if gather_shift is not None else shift
        scale = gather_scale if gather_scale is not None else scale

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states[:, :-pad_len, :] if pad_len > 0 else hidden_states
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

def WanTransformerBlockForwardGaudi (
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if temb.ndim == 4:
        # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        # batch_size, seq_len, 1, inner_dim
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)
    # 1. Self-attention
    norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = self.attn1(norm_hidden_states, None, attention_mask, rotary_emb)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output
    # 3. Feed-forward
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
        hidden_states
    )
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
    return hidden_states
