# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch XLM-RoBERTa model."""

from typing import Optional

import torch
import torch.utils.checkpoint
from transformers.cache_utils import Cache, EncoderDecoderCache
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from optimum.utils import logging


logger = logging.get_logger(__name__)


#def gaudi_XLMRoberta_Sdpa_SelfAttention_forward(
#    self,
#    hidden_states: torch.Tensor,
#    attention_mask: Optional[torch.Tensor] = None,
#    head_mask: Optional[torch.FloatTensor] = None,
#    encoder_hidden_states: Optional[torch.FloatTensor] = None,
#    past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
#    output_attentions: Optional[bool] = False,
#    cache_position: Optional[torch.Tensor] = None,
#) -> tuple[torch.Tensor]:
#    r"""
#    Copied from https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L295
#    Changes:
#        - Use HPU's FusedSDPA(fast mode for softmax) to replace `orch.nn.functional.scaled_dot_product_attention`
#    """
#    if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
#        # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
#        logger.warning_once(
#            "XLMRobertaSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
#            "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
#            "the manual attention implementation, but specifying the manual implementation will be required from "
#            "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
#            '`attn_implementation="eager"` when loading the model.'
#        )
#        return super().forward(
#            hidden_states,
#            attention_mask,
#            head_mask,
#            encoder_hidden_states,
#            past_key_value,
#            output_attentions,
#            cache_position,
#        )
#
#    bsz, tgt_len, _ = hidden_states.size()
#
#    query_layer = (
#        self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
#    )
#
#    is_cross_attention = encoder_hidden_states is not None
#
#    current_states = encoder_hidden_states if is_cross_attention else hidden_states
#
#    # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
#    if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
#        key_layer, value_layer = past_key_value
#    else:
#        key_layer = self.transpose_for_scores(self.key(current_states))
#        value_layer = self.transpose_for_scores(self.value(current_states))
#        if past_key_value is not None and not is_cross_attention:
#            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#
#    if self.is_decoder:
#        past_key_value = (key_layer, value_layer)
#
#    is_causal = self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1
#
#    attn_output = FusedSDPA.apply(
#        query_layer, key_layer, value_layer, attention_mask, 0.0, is_causal, None, "fast", False
#    )
#
#    attn_output = attn_output.transpose(1, 2)
#    attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
#
#    outputs = (attn_output,)
#    if self.is_decoder:
#        outputs = outputs + (past_key_value,)
#    return outputs

def gaudi_XLMRoberta_Sdpa_SelfAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor]:
    if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
        # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
        logger.warning_once(
            "XLMRobertaSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
            "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
            "the manual attention implementation, but specifying the manual implementation will be required from "
            "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
            '`attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            past_key_value,
            output_attentions,
            cache_position,
        )

    bsz, tgt_len, _ = hidden_states.size()

    query_layer = (
        self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
    )

    is_cross_attention = encoder_hidden_states is not None
    current_states = encoder_hidden_states if is_cross_attention else hidden_states
    if past_key_value is not None:
        if isinstance(past_key_value, EncoderDecoderCache):
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache
        else:
            curr_past_key_value = past_key_value

    current_states = encoder_hidden_states if is_cross_attention else hidden_states
    if is_cross_attention and past_key_value is not None and is_updated:
        # reuse k,v, cross_attentions
        key_layer = curr_past_key_value.layers[self.layer_idx].keys
        value_layer = curr_past_key_value.layers[self.layer_idx].values
    else:
        key_layer = (
            self.key(current_states)
            .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(current_states)
            .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        if past_key_value is not None:
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_layer, value_layer = curr_past_key_value.update(
                key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True

    # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
    # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
    # Reference: https://github.com/pytorch/pytorch/issues/112577
    if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
    # a causal mask in case tgt_len == 1.
    is_causal = self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1

    #attn_output = torch.nn.functional.scaled_dot_product_attention(
    #    query_layer,
    #    key_layer,
    #    value_layer,
    #    attn_mask=attention_mask,
    #    dropout_p=self.dropout_prob if self.training else 0.0,
    #    is_causal=is_causal,
    #)

    attn_output = FusedSDPA.apply(
        query_layer, key_layer, value_layer, attention_mask, 0.0, is_causal, None, "fast", False
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

    return attn_output, None

