
# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...distributed import parallel_state

import habana_frameworks.torch.core as htcore
import time #tmp

def HunyuanVideo15Transformer3DModelForwardGaudi(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    timestep_r: Optional[torch.LongTensor] = None,
    encoder_hidden_states_2: Optional[torch.Tensor] = None,
    encoder_attention_mask_2: Optional[torch.Tensor] = None,
    image_embeds: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_hunyuan_video15.py#L623
    add mark_step.
    !!!To do: cp
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
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb = self.time_embed(timestep, timestep_r=timestep_r)

    hidden_states = self.x_embedder(hidden_states)

    pad_len = 0
    if parallel_state.sequence_parallel_is_initialized():
        bs, seq_len, _ = hidden_states.shape
        cp_size = parallel_state.get_sequence_parallel_world_size()
        # We need to ensure seq_len can be divided by cp_size
        if seq_len % cp_size != 0:
            padded_seq_len = (seq_len // cp_size + 1) * cp_size
            pad_len = padded_seq_len - seq_len
            print("exitdebug -HunyuanVideo15Transformer3DModelForwardGaudi pad_len=",pad_len)
            exit()
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            cos = F.pad(image_rotary_emb[0], (0, 0, 0, pad_len))
            sin = F.pad(image_rotary_emb[1], (0, 0, 0, pad_len))
            image_rotary_emb = (cos, sin)
            # if timestep.ndim == 2:
            #     timestep = F.pad(timestep, (0, pad_len))

            seq_len = padded_seq_len

        sp_seq_len = seq_len // parallel_state.get_sequence_parallel_world_size()
        start = sp_seq_len * parallel_state.get_sequence_parallel_rank()
        end = sp_seq_len * (parallel_state.get_sequence_parallel_rank() + 1)

        hidden_states = hidden_states[:, start:end, :]

        # # timestep with 2 dims means expand_timesteps in config is True, and
        # # We only need to split the timestep when it has 2 dim.
        # expanded_timestep = False
        # if timestep.ndim == 2:
        #     expanded_timestep = True
        #     timestep = timestep[:, start:end]

        cos = image_rotary_emb[0][start:end, :]
        sin = image_rotary_emb[1][start:end, :]
        image_rotary_emb = (cos, sin)


    # qwen text embedding
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    encoder_hidden_states_cond_emb = self.cond_type_embed(
        torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
    )

    encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

    # byt5 text embedding
    encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

    encoder_hidden_states_2_cond_emb = self.cond_type_embed(
        torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
    )
    encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

    # image embed
    encoder_hidden_states_3 = self.image_embedder(image_embeds)
    is_t2v = torch.all(image_embeds == 0)
    if is_t2v:
        encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
        encoder_attention_mask_3 = torch.zeros(
            (batch_size, encoder_hidden_states_3.shape[1]),
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
    else:
        encoder_attention_mask_3 = torch.ones(
            (batch_size, encoder_hidden_states_3.shape[1]),
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
    encoder_hidden_states_3_cond_emb = self.cond_type_embed(
        2
        * torch.ones_like(
            encoder_hidden_states_3[:, :, 0],
            dtype=torch.long,
        )
    )
    encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

    # reorder and combine text tokens: combine valid tokens first, then padding
    encoder_attention_mask = encoder_attention_mask.bool()
    encoder_attention_mask_2 = encoder_attention_mask_2.bool()
    encoder_attention_mask_3 = encoder_attention_mask_3.bool()
    new_encoder_hidden_states = []
    new_encoder_attention_mask = []

    for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        encoder_hidden_states_3,
        encoder_attention_mask_3,
    ):
        # Concatenate: [valid_image, valid_byt5, valid_mllm, invalid_image, invalid_byt5, invalid_mllm]
        new_encoder_hidden_states.append(
            torch.cat(
                [
                    image[image_mask],  # valid image
                    text_2[text_mask_2],  # valid byt5
                    text[text_mask],  # valid mllm
                    image[~image_mask],  # invalid image
                    torch.zeros_like(text_2[~text_mask_2]),  # invalid byt5 (zeroed)
                    torch.zeros_like(text[~text_mask]),  # invalid mllm (zeroed)
                ],
                dim=0,
            )
        )

        encoder_pad_len = image_mask[~image_mask].shape[0]+text_mask_2[~text_mask_2].shape[0]+text_mask[~text_mask].shape[0]
        print("encoder_pad_len=",encoder_pad_len)

        # Apply same reordering to attention masks
        new_encoder_attention_mask.append(
            torch.cat(
                [
                    image_mask[image_mask],
                    text_mask_2[text_mask_2],
                    text_mask[text_mask],
                    image_mask[~image_mask],
                    text_mask_2[~text_mask_2],
                    text_mask[~text_mask],
                ],
                dim=0,
            )
        )


    encoder_hidden_states = torch.stack(new_encoder_hidden_states)
    encoder_attention_mask = torch.stack(new_encoder_attention_mask)

    # Do not use mask to get better perf.
    # May influent the accuracy.
    # As seqlen in hidden_states is much bigger than encoder_hidden_states,
    # Accuracy loss is acceptable. More test needed to verify.
    encoder_attention_mask =None

    htcore.mark_step()

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
                pad_len,
                encoder_pad_len,
            )
            htcore.mark_step()

    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
                pad_len,
                encoder_pad_len,
            )
            htcore.mark_step()

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
        gather1.wait()

        hidden_states = gather_hidden.reshape(bs, seq * cp_size, dim)
        hidden_states = hidden_states[:, :-pad_len, :] if pad_len > 0 else hidden_states

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)


    return Transformer2DModelOutput(sample=hidden_states)

def HunyuanVideo15TransformerBlockForwardGaudi(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    pad_len: Optional[int] = 0,
    encoder_pad_len: Optional[int] = 0,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_hunyuan_video15.py#L466
    add encoder_pad_len parameter.

    """
    # 1. Input normalization
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    # 2. Joint attention
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=freqs_cis,
        pad_len=pad_len,
        encoder_pad_len=encoder_pad_len,
    )

    # 3. Modulation and residual connection
    hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
    encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

    norm_hidden_states = self.norm2(hidden_states)
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]


    # 4. Feed-forward
    ff_output = self.ff(norm_hidden_states)
    context_ff_output = self.ff_context(norm_encoder_hidden_states)

    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output


    return hidden_states, encoder_hidden_states

def HunyuanVideo15IndividualTokenRefinerForwardGaudi(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    self_attn_mask = None
    if attention_mask is not None:
        batch_size = attention_mask.shape[0]
        seq_len = attention_mask.shape[1]
        attention_mask = attention_mask.to(hidden_states.device).bool()
        self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
        self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
        self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

    for block in self.refiner_blocks:
        hidden_states = block(hidden_states, temb, self_attn_mask)
        htcore.mark_step()

    return hidden_states

