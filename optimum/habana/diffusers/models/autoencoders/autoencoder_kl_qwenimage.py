# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team. All rights reserved.
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
#
# We gratefully acknowledge the Wan Team for their outstanding contributions.
# QwenImageVAE is further fine-tuned from the Wan Video VAE to achieve improved performance.
# For more information about the Wan VAE, please refer to:
# - GitHub: https://github.com/Wan-Video/Wan2.1
# - arXiv: https://arxiv.org/abs/2503.20314


import habana_frameworks.torch.core as htcore
import torch


CACHE_T = 2


def QwenImageEncoder3dForwardGaudi(self, x, feat_cache=None, feat_idx=[0]):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py#L440
    only add mark_step() for memory optimization and reduce compile time.
    """

    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)

    htcore.mark_step()
    ## downsamples
    for layer in self.down_blocks:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
        htcore.mark_step()

    ## middle
    x = self.mid_block(x, feat_cache, feat_idx)
    htcore.mark_step()

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    htcore.mark_step()

    return x


def QwenImageDecoder3dForwardGaudi(self, x, feat_cache=None, feat_idx=[0]):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py#L628
    only add mark_step() for memory optimization and reduce compile time.
    """

    ## conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)
    htcore.mark_step()

    ## middle
    x = self.mid_block(x, feat_cache, feat_idx)
    htcore.mark_step()

    ## upsamples
    for up_block in self.up_blocks:
        x = up_block(x, feat_cache, feat_idx)
        htcore.mark_step()

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    htcore.mark_step()
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    htcore.mark_step()
    return x


def QwenImageAttentionBlockForwardGaudi(self, x):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py#L306
    Replace scaled_dot_product_attention with Gaudi's FusedSDPA and add mark_step()
    """
    identity = x
    batch_size, channels, time, height, width = x.size()

    x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
    x = self.norm(x)

    # compute query, key, value
    qkv = self.to_qkv(x)
    qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
    qkv = qkv.permute(0, 1, 3, 2).contiguous()
    q, k, v = qkv.chunk(3, dim=-1)

    # apply attention
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    x = FusedSDPA.apply(q, k, v, None, 0.0, False, None, "None", None)
    htcore.mark_step()

    x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

    # output projection
    x = self.proj(x)

    # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
    x = x.view(batch_size, time, channels, height, width)
    x = x.permute(0, 2, 1, 3, 4)

    return x + identity
