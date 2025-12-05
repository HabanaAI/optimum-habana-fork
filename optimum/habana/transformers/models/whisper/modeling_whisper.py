from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.whisper.modeling_whisper import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperEncoderLayer,
    WhisperForConditionalGeneration,
    WhisperModel,
    eager_attention_forward,
    shift_tokens_right,
)
from transformers.utils import logging
from typing_extensions import Unpack


logger = logging.get_logger(__name__)


class GaudiWhisperAttention(WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self.training and getattr(self, "gradient_checkpointing", False):
            past_key_value = None

        is_cross_attention = key_value_states is not None
        bsz, tgt_len = hidden_states.shape[:-1]

        # Compute queries
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # Handle cross/self attention caching
        if past_key_value is not None and isinstance(past_key_value, EncoderDecoderCache):
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states

        if is_cross_attention and past_key_value and is_updated:
            key_states = past_key_value.layers[self.layer_idx].keys
            value_states = past_key_value.layers[self.layer_idx].values
        else:
            key_states = (
                self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            )
            value_states = (
                self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            )
            if past_key_value is not None:
                cache_position = None if is_cross_attention else cache_position
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.dropout if self.training else 0.0,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class GaudiWhisperDecoderLayer(WhisperDecoderLayer):
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, "_gradient_checkpointing_func"):
            delattr(self, "_gradient_checkpointing_func")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
            token_idx=token_idx,
        )
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            pkv_cross = past_key_value if use_cache else None

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=pkv_cross,
                output_attentions=output_attentions,
            )
            hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class GaudiWhisperDecoder(WhisperDecoder):
    supports_gradient_checkpointing = False
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = False
        for layer in self.layers:
            layer.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        token_idx=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify exactly one of decoder_input_ids or decoder_inputs_embeds (both None).")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You must specify exactly one of decoder_input_ids or decoder_inputs_embeds (both provided)."
            )

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]

        hidden_states = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)

        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(), DynamicCache())
                if self.config.is_encoder_decoder
                else DynamicCache()
            )

        past_kv_len = (
            cache_position[0]
            if cache_position is not None
            else past_key_values.get_seq_length()
            if past_key_values is not None
            else 0
        )

        if cache_position is None:
            cache_position = torch.arange(
                past_kv_len,
                past_kv_len + input_shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = (
                (token_idx - 1).unsqueeze(0)
                if token_idx is not None
                else cache_position.unsqueeze(0).repeat(input_shape[0], 1)
            )

        positions = self.embed_positions(
            input_ids if input_ids is not None else inputs_embeds,
            past_key_values_length=past_kv_len,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + positions.to(inputs_embeds.device)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training and torch.rand([]) < self.layerdrop:
                continue

            layer_head = head_mask[idx] if head_mask is not None else None
            cross_layer_head = cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                layer_head_mask=layer_head,
                cross_attn_layer_head_mask=cross_layer_head,
                past_key_value=past_key_values if use_cache else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                token_idx=token_idx,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attns += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attns]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attns,
        )



class GaudiWhisperEncoderLayer(WhisperEncoderLayer):
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, "_gradient_checkpointing_func"):
            delattr(self, "_gradient_checkpointing_func")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.FloatTensor]:
        if self.training and getattr(self, "gradient_checkpointing", False):

            def custom_forward(hidden_states, attention_mask, layer_head_mask):
                return super(GaudiWhisperEncoderLayer, self).forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=layer_head_mask,
                    output_attentions=False,
                )

            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                layer_head_mask,
                use_reentrant=False,
            )

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )


class GaudiWhisperEncoder(WhisperEncoder):
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)

        new_layers = nn.ModuleList()
        for layer in self.layers:
            gaudi_layer = GaudiWhisperEncoderLayer(config)
            gaudi_layer.load_state_dict(layer.state_dict())
            new_layers.append(gaudi_layer)

        self.layers = new_layers

    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, "
                f"but found {input_features.shape[-1]}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.gradient_checkpointing:
            output_attentions = False

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers)

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            to_drop = False
            if self.training:
                if torch.rand([]) < self.layerdrop:
                    to_drop = True

            if to_drop:
                layer_outputs = (hidden_states, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in (hidden_states, encoder_states, all_attentions) if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class GaudiWhisperModel(WhisperModel):
    supports_gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        # disable for encoder layers
        if isinstance(module, GaudiWhisperEncoderLayer):
            module.gradient_checkpointing = False
            return
        
        # disable for decoder layers
        if isinstance(module, GaudiWhisperDecoderLayer):
            module.gradient_checkpointing = False
            return

        # disable for encoder/decoder modules
        if isinstance(module, (GaudiWhisperEncoder, GaudiWhisperDecoder)):
            module.gradient_checkpointing = False
            return

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[Cache]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.decoder.gradient_checkpointing:
            use_cache = False
            past_key_values = None

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            token_idx=token_idx,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class GaudiWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    supports_gradient_checkpointing = False
    
    def __init__(self, config):
        super().__init__(config)
        self._supports_cache_class = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}

        self._gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

        if hasattr(self.model, "encoder"):
            encoder = self.model.encoder
            encoder.gradient_checkpointing = True
            for layer in getattr(encoder, "layers", []):
                setattr(layer, "gradient_checkpointing", True)

        if hasattr(self.model, "decoder"):
            decoder = self.model.decoder
            decoder.gradient_checkpointing = False
            for layer in getattr(decoder, "layers", []):
                if hasattr(layer, "gradient_checkpointing"):
                    setattr(layer, "gradient_checkpointing", False)

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}

        if hasattr(self.model, "encoder"):
            encoder = self.model.encoder
            encoder.gradient_checkpointing = False
            for layer in getattr(encoder, "layers", []):
                if hasattr(layer, "gradient_checkpointing"):
                    setattr(layer, "gradient_checkpointing", False)

        if hasattr(self.model, "decoder"):
            decoder = self.model.decoder
            decoder.gradient_checkpointing = False
            for layer in getattr(decoder, "layers", []):
                if hasattr(layer, "gradient_checkpointing"):
                    setattr(layer, "gradient_checkpointing", False)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[Cache]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if token_idx is not None:
            if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

            outputs = self.model(
                input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_position_ids=decoder_position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                token_idx=token_idx,
            )

            lm_logits = self.proj_out(outputs[0])
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                labels = labels.to(lm_logits.device)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        return super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
