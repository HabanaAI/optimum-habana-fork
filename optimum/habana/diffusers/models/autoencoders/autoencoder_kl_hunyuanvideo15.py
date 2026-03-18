import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import FusedSDPA

from ....distributed import parallel_state


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
    _, _, _, height, width = z.shape

    overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
    overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor))  # 8 * (1 - 0.25) = 6
    blend_height = int(self.tile_sample_min_height * self.tile_overlap_factor)  # 256 * 0.25 = 64
    blend_width = int(self.tile_sample_min_width * self.tile_overlap_factor)  # 256 * 0.25 = 64
    row_limit_height = self.tile_sample_min_height - blend_height  # 256 - 64 = 192
    row_limit_width = self.tile_sample_min_width - blend_width  # 256 - 64 = 192

    htcore.mark_step()
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
            ].clone()
            decoded = self.decoder(tile)
            row.append(decoded)
            htcore.mark_step()
        rows.append(row)
        htcore.mark_step()


    htcore.mark_step()
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):

            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_height)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_width)

            result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width].clone())
            htcore.mark_step()
        result_rows.append(torch.cat(result_row, dim=-1))
        htcore.mark_step()
    dec = torch.cat(result_rows, dim=-2)

    return dec

def prepare_causal_attention_mask_bool(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    """Prepare a causal attention mask for 3D videos.

    Args:
        n_frame (int): Number of frames (temporal length).
        n_hw (int): Product of height and width.
        dtype: Desired mask dtype.
        device: Device for the mask.
        batch_size (int, optional): If set, expands for batch.

    Returns:
        torch.Tensor: Causal attention mask.
    """
    seq_len = n_frame * n_hw
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 1#0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

def prepare_causal_attention_mask_optimized(n_frame: int, n_hw: int, device, batch_size: int = None):
    seq_len = n_frame * n_hw

    # 创建帧索引 [0, 0, ..., 1, 1, ..., n_frame-1]
    # 形状为 (seq_len,)
    frame_idx = torch.arange(seq_len, device=device) // n_hw

    # 广播比较: [seq_len, 1] >= [1, seq_len]
    # 逻辑：如果 Query 的帧索引 >= Key 的帧索引，则可见
    mask = frame_idx.unsqueeze(1) >= frame_idx.unsqueeze(0)

    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    return mask # 返回的就是 torch.bool

def HunyuanVideo15AttnBlockForwardGaudi(self, x: torch.Tensor) -> torch.Tensor:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanvideo15.py#L129
    Replace scaled_dot_product_attention with Gaudi's FusedSDPA and add mark_step()
    Optimize prepare_causal_attention_mask with Bool mask instead of float
    """

    identity = x

    x = self.norm(x)

    query = self.to_q(x)
    key = self.to_k(x)
    value = self.to_v(x)

    batch_size, channels, frames, height, width = query.shape

    query = query.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
    key = key.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()
    value = value.reshape(batch_size, channels, frames * height * width).permute(0, 2, 1).unsqueeze(1).contiguous()

    attention_mask = prepare_causal_attention_mask_optimized(
        frames, height * width, query.device, batch_size=batch_size
    )

    htcore.mark_step()

    x = FusedSDPA.apply(query, key, value, attention_mask, 0.0, False, None, "fast", None)
    htcore.mark_step()

    # batch_size, 1, frames * height * width, channels
    x = x.squeeze(1).reshape(batch_size, frames, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()
    htcore.mark_step()
    x = self.proj_out(x)

    return x + identity


def HunyuanVideo15CausalConv3dForwardGaudi(self, hidden_states: torch.Tensor) -> torch.Tensor:

        if parallel_state.sequence_parallel_is_initialized():
            htcore.mark_step()
            # for multi-cards case, pad_mode=replicate,takes long time.
            # Using expand + cat , to replace "replicate" padding to reduce hpu process time.
            # when kernel_size=3, self.time_causal_padding= (1, 1, 1, 1, 2, 0)
            # 1. Pad Time (dim 2): Prepend the first frame 2 times
            t_pad = hidden_states[:, :, :1, :, :].expand(-1, -1, self.time_causal_padding[4], -1, -1)
            hidden_states = torch.cat([t_pad, hidden_states], dim=2)

            # 2. Pad Height (dim 3): Prepend first row and append last row
            h_top = hidden_states[:, :, :, :1, :].expand(-1, -1, -1, self.time_causal_padding[2], -1)
            h_bottom = hidden_states[:, :, :, -1:, :].expand(-1, -1, -1, self.time_causal_padding[3], -1)
            hidden_states = torch.cat([h_top, hidden_states, h_bottom], dim=3)

            # 3. Pad Width (dim 4): Prepend first column and append last column
            w_left = hidden_states[:, :, :, :, :1].expand(-1, -1, -1, -1, self.time_causal_padding[0])
            w_right = hidden_states[:, :, :, :, -1:].expand(-1, -1, -1, -1, self.time_causal_padding[1])
            hidden_states = torch.cat([w_left, hidden_states, w_right], dim=4)
            htcore.mark_step()
        else:
            hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)

        htcore.mark_step()

        return self.conv(hidden_states)


def HunyuanVideo15UpBlock3DForwardGaudi(self, hidden_states: torch.Tensor) -> torch.Tensor:
    r"""
    add mark_step()
    """

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for resnet in self.resnets:
            hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)

    else:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
            htcore.mark_step()

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
            htcore.mark_step()

    return hidden_states


def HunyuanVideo15Decoder3DForwardGaudi(self, hidden_states: torch.Tensor) -> torch.Tensor:

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

    htcore.mark_step()
    hidden_states = self.conv_out(hidden_states)

    return hidden_states
