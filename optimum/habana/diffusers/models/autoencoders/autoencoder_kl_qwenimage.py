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

from typing import Optional, Union

import torch
import habana_frameworks.torch.core as htcore

CACHE_T = 2

def QwenImageEncoder3dForwardGaudi(self, x, feat_cache=None, feat_idx=[0]):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py#L442
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
    Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py#L633
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