import torch
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels import FusedSDPA

def AutoencoderKLHunyuanVideo15TiledDecodeGaudi(self, z: torch.Tensor) -> torch.Tensor:
    r"""
    Decode a batch of images using a tiled decoder.

    Args:
        z (`torch.Tensor`): Input batch of latent vectors.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

    Returns:
        [`~models.vae.DecoderOutput`] or `tuple`:
            If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
            returned.
    """
    print("~~~~~~AutoencoderKLHunyuanVideo15_tiled_decode_Gaudi~)))))))))))))~~~~")
    _, _, _, height, width = z.shape

    overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
    overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
    blend_height = int(self.tile_sample_min_height * self.tile_overlap_factor)  # 256 * 0.25 = 64
    blend_width = int(self.tile_sample_min_width * self.tile_overlap_factor)  # 256 * 0.25 = 64
    row_limit_height = self.tile_sample_min_height - blend_height  # 256 - 64 = 192
    row_limit_width = self.tile_sample_min_width - blend_width  # 256 - 64 = 192

    rows = []
    for i in range(0, height, overlap_height):
        row = []
        for j in range(0, width, overlap_width):
            tile = z[
                :,
                :,
                :,
                i : i + self.tile_latent_min_height,
                j : j + self.tile_latent_min_width,
            ]
            decoded = self.decoder(tile)
            row.append(decoded)
            htcore.mark_step()
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_height)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            htcore.mark_step()
        result_rows.append(torch.cat(result_row, dim=-1))
    dec = torch.cat(result_rows, dim=-2)

    return dec


def HunyuanVideo15AttnBlockForwardGaudi(self, x: torch.Tensor) -> torch.Tensor:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanvideo15.py#L129
    Replace scaled_dot_product_attention with Gaudi's FusedSDPA and add mark_step()
    """
    print("~~~~~~HunyuanVideo15AttnBlockForwardGaudi~++++++++++~~~~")

    identity = x

    x = self.norm(x)

    query = self.to_q(x)
    key = self.to_k(x)
    value = self.to_v(x)

    batch_size, channels, frames, height, width = query.shape

    query = query.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
    key = key.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
    value = value.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()

    attention_mask = self.prepare_causal_attention_mask(
        frames, height * width, query.dtype, query.device, batch_size=batch_size
    )

    x = FusedSDPA.apply(query, key, value, attention_mask, 0.0, False, None, "None", None)
    htcore.mark_step()

    # batch_size, 1, frames * height * width, channels

    x = x.squeeze(1).reshape(batch_size, frames, height, width, channels).permute(0, 4, 1, 2, 3)
    x = self.proj_out(x)

    return x + identity

def HunyuanVideo15Decoder3DForwardGaudi(self, hidden_states: torch.Tensor) -> torch.Tensor:

    print("~~~~~~HunyuanVideo15Decoder3DForwardGaudi~~~~~")
    hidden_states = self.conv_in(hidden_states) + hidden_states.repeat_interleave(repeats=self.repeat, dim=1)
    htcore.mark_step()

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)

        for up_block in self.up_blocks:
            hidden_states = self._gradient_checkpointing_func(up_block, hidden_states)

    else:
        hidden_states = self.mid_block(hidden_states)

        htcore.mark_step()
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)
        htcore.mark_step()

    # post-process
    hidden_states = self.norm_out(hidden_states)
    hidden_states = self.conv_act(hidden_states)
    hidden_states = self.conv_out(hidden_states)

    return hidden_states