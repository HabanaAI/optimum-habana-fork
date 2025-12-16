# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingMode, apply_rotary_pos_emb

from ...distributed import parallel_state


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = torch.repeat_interleave(cos.unsqueeze(1), 2, dim=2, output_size=128)
        sin = torch.repeat_interleave(sin.unsqueeze(1), 2, dim=2, output_size=128)
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def apply_rotary_emb_qwen_gaudi(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis  # [S, D]
    cos = torch.repeat_interleave(cos.unsqueeze(1), 2, dim=2, output_size=128)
    sin = torch.repeat_interleave(sin.unsqueeze(1), 2, dim=2, output_size=128)

    ori_dtype = x.dtype
    x = x.to(cos.dtype)

    out = apply_rotary_pos_emb(x, cos, sin, None, 0, RotaryPosEmbeddingMode.PAIRWISE)

    return out.to(ori_dtype)


def QwenImageTransformer2DModelGaudi(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    return_dict: bool = True,
    hidden_states_pad_len: int = 0,
    encoder_hidden_states_pad_len: int = 0,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_qwenimage.py#L479
    Add mark_step.
    replace rope complex computation to real.
    Add cp support.
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
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.img_in(hidden_states)

    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    (vid_freqs_cos, vid_freqs_sin), (txt_freqs_cos, txt_freqs_sin) = self.pos_embed(
        img_shapes, txt_seq_lens, device=hidden_states.device
    )

    if hidden_states_pad_len > 0:
        vid_freqs_cos = F.pad(vid_freqs_cos, (0, 0, 0, hidden_states_pad_len))
        vid_freqs_sin = F.pad(vid_freqs_sin, (0, 0, 0, hidden_states_pad_len))
    if encoder_hidden_states_pad_len > 0:
        txt_freqs_cos = F.pad(txt_freqs_cos, (0, 0, 0, encoder_hidden_states_pad_len))
        txt_freqs_sin = F.pad(txt_freqs_sin, (0, 0, 0, encoder_hidden_states_pad_len))

    pad_len_img = 0
    attention_mask = None
    if parallel_state.sequence_parallel_is_initialized():
        bs, seq_len_img, _ = hidden_states.shape

        cp_size = parallel_state.get_sequence_parallel_world_size()
        # We need to ensure seq_len can be divided by cp_size
        if seq_len_img % cp_size != 0:
            padded_seq_len_img = (seq_len_img // cp_size + 1) * cp_size
            pad_len_img = padded_seq_len_img - seq_len_img
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len_img))

            vid_freqs_cos = F.pad(vid_freqs_cos, (0, 0, 0, pad_len_img))
            vid_freqs_sin = F.pad(vid_freqs_sin, (0, 0, 0, pad_len_img))

            seq_len_img = padded_seq_len_img

        sp_seq_len_img = seq_len_img // parallel_state.get_sequence_parallel_world_size()
        start_img = sp_seq_len_img * parallel_state.get_sequence_parallel_rank()
        end_img = sp_seq_len_img * (parallel_state.get_sequence_parallel_rank() + 1)
        hidden_states = hidden_states[:, start_img:end_img, :]

        vid_freqs_cos = vid_freqs_cos[start_img:end_img, :]
        vid_freqs_sin = vid_freqs_sin[start_img:end_img, :]

    image_rotary_emb = (vid_freqs_cos, vid_freqs_sin), (txt_freqs_cos, txt_freqs_sin)

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
                attention_mask,
                encoder_hidden_states_pad_len,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                attention_mask=attention_mask,
                encoder_hidden_states_pad_len=encoder_hidden_states_pad_len,
            )

        htcore.mark_step()

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

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

        hidden_states = hidden_states[:, :-pad_len_img, :] if pad_len_img > 0 else hidden_states

    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def QwenImageTransformerBlockForwardGaudi(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states_pad_len: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapted from https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_qwenimage.py#L411
    Add attention_mask.
    """
    # Get modulation parameters for both streams
    img_mod_params = self.img_mod(temb)  # [B, 6*dim]
    txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

    # Split modulation parameters for norm1 and norm2
    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

    # Process image stream - norm1 + modulation
    img_normed = self.img_norm1(hidden_states)
    img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

    # Process text stream - norm1 + modulation
    txt_normed = self.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

    # Use QwenAttnProcessor2_0 for joint attention computation
    # This directly implements the DoubleStreamLayerMegatron logic:
    # 1. Computes QKV for both streams
    # 2. Applies QK normalization and RoPE
    # 3. Concatenates and runs joint attention
    # 4. Splits results back to separate streams
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=img_modulated,  # Image stream (will be processed as "sample")
        encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
        attention_mask=attention_mask,
        encoder_hidden_states_pad_len=encoder_hidden_states_pad_len,
        **joint_attention_kwargs,
    )

    # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
    img_attn_output, txt_attn_output = attn_output

    # Apply attention gates and add residual (like in Megatron)
    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    # Process image stream - norm2 + MLP
    img_normed2 = self.img_norm2(hidden_states)
    img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
    img_mlp_output = self.img_mlp(img_modulated2)
    hidden_states = hidden_states + img_gate2 * img_mlp_output

    # Process text stream - norm2 + MLP
    txt_normed2 = self.txt_norm2(encoder_hidden_states)
    txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
    txt_mlp_output = self.txt_mlp(txt_modulated2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

    # Clip to prevent overflow for fp16
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states
