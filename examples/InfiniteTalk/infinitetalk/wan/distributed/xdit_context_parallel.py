# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp

from wan.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from einops import rearrange

from ..modules.model import sinusoidal_embedding_1d
from ..utils.multitalk_utils import get_attn_map_with_target, split_token_counts_and_frame_ids, normalize_and_scale
from ..modules.attention import SingleStreamAttention, SingleStreamMutiAttention, attention, FlashAttnV3Gaudi

import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingMode, apply_rotary_pos_emb


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3], dim=1
    )  # [[N, head_dim/2], [N, head_dim/2], [N, head_dim/2]] # T H W 极坐标

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))  # [L, N, C/2] # 极坐标
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)  # seq_lens, 1,  3 * dim / 2 (T H W)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@amp.autocast(enabled=False)
def rope_apply_gaudi(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    cos, sin = freqs
    cos = cos.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    sin = sin.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = x.reshape(s, n, -1, 2)
        x_real, x_imag = x_i.unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

        cos_i = torch.cat(
            [
                cos[0][:f].reshape(f, 1, 1, -1).expand(f, h, w, -1),
                cos[1][:h].reshape(1, h, 1, -1).expand(f, h, w, -1),
                cos[2][:w].reshape(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        sin_i = torch.cat(
            [
                sin[0][:f].reshape(f, 1, 1, -1).expand(f, h, w, -1),
                sin[1][:h].reshape(1, h, 1, -1).expand(f, h, w, -1),
                sin[2][:w].reshape(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        cos_i = cos_i.to(device=x.device)
        sin_i = sin_i.to(device=x.device)

        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        cos_i = pad_freqs(cos_i, s * sp_size)
        sin_i = pad_freqs(sin_i, s * sp_size)

        s_per_rank = s
        cos_i_rank = cos_i[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :]
        sin_i_rank = sin_i[(sp_rank * s_per_rank) : ((sp_rank + 1) * s_per_rank), :, :]

        cos_i_rank = torch.repeat_interleave(cos_i_rank, 2, dim=2).reshape(s, 1, -1, 2)
        sin_i_rank = torch.repeat_interleave(sin_i_rank, 2, dim=2).reshape(s, 1, -1, 2)

        x_i = (x_i.float() * cos_i_rank + x_rotated.float() * sin_i_rank).flatten(2).to(x.dtype)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output)


def usp_dit_forward_vace(self, x, vace_context, seq_len, kwargs):
    # embeddings
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    c = [u.flatten(2).transpose(1, 2) for u in c]
    c = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in c])

    # arguments
    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)

    # Context Parallel
    c = torch.chunk(c, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    hints = []
    for block in self.vace_blocks:
        c, c_skip = block(c, **new_kwargs)
        hints.append(c_skip)
    return hints


def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    vace_context=None,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device

    if self.model_type != "vace" and y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    )

    if self.model_type != "vace" and clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens
    )

    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def usp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = attention(
        query=half(q),
        key=half(k),
        value=half(v),
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_dit_forward_multitalk(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    audio=None,
    ref_target_masks=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """

    assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device

    _, T, H, W = x[0].shape
    N_t = T // self.patch_size[0]
    N_h = H // self.patch_size[1]
    N_w = W // self.patch_size[2]

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    x[0] = x[0].to(context[0].dtype)

    freqs = self.rope(x[0])

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    # time embeddings
    with torch.autocast(device_type="hpu", dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    # get audio token
    audio_cond = audio.to(device=x.device, dtype=x.dtype)
    first_frame_audio_emb_s = audio_cond[:, :1, ...]
    latter_frame_audio_emb = audio_cond[:, 1:, ...]
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
    middle_index = self.audio_window // 2
    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, : middle_index + 1, ...]
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index : middle_index + 1, ...]
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_frame_audio_emb_s = torch.concat(
        [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2
    )
    audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
    human_num = len(audio_embedding)
    audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

    # convert ref_target_masks to token_ref_target_masks
    if ref_target_masks is not None:
        ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32)
        token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode="nearest")
        token_ref_target_masks = token_ref_target_masks.squeeze(0)
        token_ref_target_masks = token_ref_target_masks > 0
        token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1)
        token_ref_target_masks = token_ref_target_masks.to(x.dtype)

    if self.enable_teacache:
        modulated_inp = e0 if self.use_ret_steps else e
        if self.cnt % 3 == 0:  # cond
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_cond = True
                self.accumulated_rel_l1_distance_cond = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_cond += rescale_func(
                    ((modulated_inp - self.previous_e0_cond).abs().mean() / self.previous_e0_cond.abs().mean())
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_cond < self.teacache_thresh:
                    should_calc_cond = False
                else:
                    should_calc_cond = True
                    self.accumulated_rel_l1_distance_cond = 0
            self.previous_e0_cond = modulated_inp.clone()
        elif self.cnt % 3 == 1:  # drop_text
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_drop_text = True
                self.accumulated_rel_l1_distance_drop_text = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_drop_text += rescale_func(
                    (
                        (modulated_inp - self.previous_e0_drop_text).abs().mean()
                        / self.previous_e0_drop_text.abs().mean()
                    )
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_drop_text < self.teacache_thresh:
                    should_calc_drop_text = False
                else:
                    should_calc_drop_text = True
                    self.accumulated_rel_l1_distance_drop_text = 0
            self.previous_e0_drop_text = modulated_inp.clone()
        else:  # uncond
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_uncond = True
                self.accumulated_rel_l1_distance_uncond = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_uncond += rescale_func(
                    ((modulated_inp - self.previous_e0_uncond).abs().mean() / self.previous_e0_uncond.abs().mean())
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_uncond < self.teacache_thresh:
                    should_calc_uncond = False
                else:
                    should_calc_uncond = True
                    self.accumulated_rel_l1_distance_uncond = 0
            self.previous_e0_uncond = modulated_inp.clone()

    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    cos = torch.chunk(freqs[0], get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    sin = torch.chunk(freqs[1], get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    freqs = (cos, sin)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_lens=context_lens,
        audio_embedding=audio_embedding,
        ref_target_masks=token_ref_target_masks,
        human_num=human_num,
    )

    if self.enable_teacache:
        if self.cnt % 3 == 0:
            if not should_calc_cond:
                x += self.previous_residual_cond
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_cond = x - ori_x
        elif self.cnt % 3 == 1:
            if not should_calc_drop_text:
                x += self.previous_residual_drop_text
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_drop_text = x - ori_x
        else:
            if not should_calc_uncond:
                x += self.previous_residual_uncond
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_uncond = x - ori_x
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    if self.enable_teacache:
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0

    return torch.stack(x).float()


def usp_attn_forward_multitalk(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, ref_target_masks=None):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    # RoPE
    q = apply_rotary_pos_emb(q, *freqs, None, 0, RotaryPosEmbeddingMode.PAIRWISE)
    k = apply_rotary_pos_emb(k, *freqs, None, 0, RotaryPosEmbeddingMode.PAIRWISE)

    # Context Parallel
    k = get_sp_group().all_gather(k, dim=1)
    v = get_sp_group().all_gather(v, dim=1)

    x = self.fav3.forward(
        half(q), half(k), half(v), cp_size=get_sequence_parallel_world_size(), layout_head_first=False
    )
    htcore.mark_step()

    # output
    x = x.flatten(2)
    x = self.o(x)

    x_ref_attn_map = None

    return x, x_ref_attn_map


def sp_crossattn_multi_forward(
    self,
    x: torch.Tensor,
    encoder_hidden_states: torch.Tensor, # [B, N_t, N_a, C]
    shape=None,
    x_ref_attn_map=None,
    human_num=None,
) -> torch.Tensor:
    assert human_num == 1, "sp_crossattn_multi_forward only supports human_num == 1"
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    encoder_hidden_states = encoder_hidden_states.squeeze(0) # [N_t, N_a, C]

    N_t, N_h, N_w = shape
    q = self.q_linear(x)

    B, N, C = x.shape
    q_shape = (B, N, self.num_heads, self.head_dim)  # [B, N_t*N_h*N_w/sp_size, H, D]
    q = q.reshape(q_shape)
    q = get_sp_group().all_to_all(q, scatter_dim=2, gather_dim=1)  # [B, N_t*N_h*N_w, H/sp_size, D]
    q = rearrange(q, "B (N_t S) H D -> (B N_t) S H D", N_t=N_t)

    if self.qk_norm:
        q = self.q_norm(q)

    # get kv from encoder_hidden_states
    _, N_a, _ = encoder_hidden_states.shape # N_a = audio_tokens_per_frame(32) * human_num
    encoder_k = (
        torch.matmul(encoder_hidden_states, torch.chunk(self.kv_linear.weight.T, sp_size * 2, dim=1)[sp_rank])
        + torch.chunk(self.kv_linear.bias, sp_size * 2)[sp_rank]
    )  # bias
    encoder_v = (
        torch.matmul(
            encoder_hidden_states, torch.chunk(self.kv_linear.weight.T, sp_size * 2, dim=1)[sp_rank + sp_size]
        )
        + torch.chunk(self.kv_linear.bias, sp_size * 2)[sp_rank + sp_size]
    )

    encoder_kv_shape = (N_t, N_a, self.num_heads // sp_size, self.head_dim)
    encoder_k = encoder_k.reshape(encoder_kv_shape)
    encoder_v = encoder_v.reshape(encoder_kv_shape)

    if self.qk_norm:
        encoder_k = self.add_k_norm(encoder_k)

    # compute attention
    x = self.fav3.forward(q, encoder_k, encoder_v, layout_head_first=False) # [B, N, H/sp_size, D]
    htcore.mark_step()
    # reshape x for all_to_all
    x = x.reshape((B, -1, self.num_heads // sp_size, self.head_dim)) # [B, N_t*N_h*N_w, H/sp_size, D]
    x = get_sp_group().all_to_all(x, scatter_dim=1, gather_dim=2) # [B, N_t*N_h*N_w/sp_size, H, D]

    # linear transform
    x_output_shape = (B, N, -1)
    x = x.reshape(x_output_shape)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


def usp_crossattn_multi_forward_multitalk(
    self,
    x: torch.Tensor,
    encoder_hidden_states: torch.Tensor,  # 1, 21, 64, C
    shape=None,
    x_ref_attn_map=None,
    human_num=None,
) -> torch.Tensor:
    N_t, N_h, N_w = shape
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    audio_tokens_per_frame = 32
    visual_seqlen, frame_ids = split_token_counts_and_frame_ids(N_t, N_h * N_w, sp_size, sp_rank)
    encoder_hidden_states = encoder_hidden_states[:, min(frame_ids) : max(frame_ids) + 1, ...]
    encoder_hidden_states = rearrange(encoder_hidden_states, "B T N C -> B (T N) C")
    N_a = len(frame_ids)
    kv_seq = [audio_tokens_per_frame * human_num] * N_a

    if human_num == 1:
        return super(SingleStreamMutiAttention, self).forward(
            x, encoder_hidden_states, shape, enable_sp=True, kv_seq=kv_seq
        )

    # get q for hidden_state
    B, N, C = x.shape
    q = self.q_linear(x)
    q_shape = (B, N, self.num_heads, self.head_dim)
    q = q.view(q_shape).permute((0, 2, 1, 3))

    if self.qk_norm:
        q = self.q_norm(q)

    max_values = x_ref_attn_map.max(1).values[:, None, None]
    min_values = x_ref_attn_map.min(1).values[:, None, None]
    max_min_values = torch.cat([max_values, min_values], dim=2)
    max_min_values = get_sp_group().all_gather(max_min_values, dim=1)

    human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
    human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

    human1 = normalize_and_scale(
        x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1])
    )
    human2 = normalize_and_scale(
        x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1])
    )
    back = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
    max_indices = x_ref_attn_map.argmax(dim=0)
    normalized_map = torch.stack([human1, human2, back], dim=1)
    normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices]  # N
    q = self.rope_1d(q, normalized_pos)

    encoder_kv = self.kv_linear(encoder_hidden_states)
    encoder_kv_shape = (B, encoder_hidden_states.size(1), 2, self.num_heads, self.head_dim)
    encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4))
    encoder_k, encoder_v = encoder_kv.unbind(0)  # B H N C

    if self.qk_norm:
        encoder_k = self.add_k_norm(encoder_k)

    # position embedding for condition audio embeddings
    per_frame = torch.zeros(audio_tokens_per_frame * human_num, dtype=encoder_k.dtype).to(encoder_k.device)
    per_frame[:audio_tokens_per_frame] = (self.rope_h1[0] + self.rope_h1[1]) / 2
    per_frame[audio_tokens_per_frame:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
    encoder_pos = torch.concat([per_frame] * N_a, dim=0)
    encoder_k = self.rope_1d(encoder_k, encoder_pos)

    # get attn
    x = self.fav3.forward(q, encoder_k, encoder_v, cp_size=get_sequence_parallel_world_size(), layout_head_first=True)
    htcore.mark_step()

    # linear transform
    x_output_shape = (B, N, C)
    x = x.transpose(1, 2)
    x = x.reshape(x_output_shape)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x
