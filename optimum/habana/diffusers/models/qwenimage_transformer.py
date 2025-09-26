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
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

import habana_frameworks.torch.core as htcore
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def apply_rotary_emb_qwen_real(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
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
    #####original
    # x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis = freqs_cis.unsqueeze(1)
    # x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    
    #####to real calculation
    x_ = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_[..., 0]
    x_imag = x_[..., 1]

    freqs_cis_real = freqs_cis.real.unsqueeze(1).to(x.device) 
    freqs_cis_imag = freqs_cis.imag.unsqueeze(1).to(x.device) 
    # freqs_cis_real = freqs_cis[0].unsqueeze(1).to(x.device) 
    # freqs_cis_imag = freqs_cis[1].unsqueeze(1).to(x.device) 

    real_part_product = x_real * freqs_cis_real - x_imag * freqs_cis_imag
    imag_part_product = x_real * freqs_cis_imag + x_imag * freqs_cis_real

    x_out = torch.stack((real_part_product, imag_part_product), dim=-1).flatten(3)  
    
    return x_out.type_as(x)

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
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
    r"""
    Adapted from: hAdapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py#L546
    Put self.pos_embed() complex type calculation on cpu.
    Add mark_step.
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

    #image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device="cpu")

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
            )
            htcore.mark_step()

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )
            htcore.mark_step()

    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)