# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

import inspect
import functools
from typing import Any, Callable, Dict, List, Optional, Union
import types
import numpy as np
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
)

from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)

from ..pipeline_utils import GaudiDiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline,calculate_shift,retrieve_timesteps
from ....transformers.gaudi_configuration import GaudiConfig
from ...models.attention_processor import GaudiQwenDoubleStreamAttnProcessor2_0
from ...models.qwenimage_transformer import QwenImageTransformer2DModelGaudi
from ....utils import HabanaProfile

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from optimum.habana.diffusers import GaudiQwenImagePipeline

        >>> pipe = GaudiQwenImagePipeline.from_pretrained(       
        ...    "Qwen/Qwen-Image",
        ...     torch_dtype=torch.bfloat16,
        ...     use_habana=True,
        ...     use_hpu_graphs=True,
        ...     gaudi_config="Habana/stable-diffusion",
        ... )

        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=50).images[0]
        >>> image.save("qwenimage.png")
        ```
"""
class GaudiQwenEmbedRope(torch.nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        # Get cos/sin components for positive indices
        pos_cos_0, pos_sin_0 = self.rope_params(pos_index, self.axes_dim[0], self.theta)
        pos_cos_1, pos_sin_1 = self.rope_params(pos_index, self.axes_dim[1], self.theta)
        pos_cos_2, pos_sin_2 = self.rope_params(pos_index, self.axes_dim[2], self.theta)

        # Get cos/sin components for negative indices
        neg_cos_0, neg_sin_0 = self.rope_params(neg_index, self.axes_dim[0], self.theta)
        neg_cos_1, neg_sin_1 = self.rope_params(neg_index, self.axes_dim[1], self.theta)
        neg_cos_2, neg_sin_2 = self.rope_params(neg_index, self.axes_dim[2], self.theta)

        # Concatenate cos components
        self.pos_freqs_cos = torch.cat([pos_cos_0, pos_cos_1, pos_cos_2], dim=1)
        self.neg_freqs_cos = torch.cat([neg_cos_0, neg_cos_1, neg_cos_2], dim=1)

        # Concatenate sin components
        self.pos_freqs_sin = torch.cat([pos_sin_0, pos_sin_1, pos_sin_2], dim=1)
        self.neg_freqs_sin = torch.cat([neg_sin_0, neg_sin_1, neg_sin_2], dim=1)

        self.rope_cache = {}
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        return cos_freqs, sin_freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs_cos.device != device:
            self.pos_freqs_cos = self.pos_freqs_cos.to(device)
            self.neg_freqs_cos = self.neg_freqs_cos.to(device)
            self.pos_freqs_sin = self.pos_freqs_sin.to(device)
            self.neg_freqs_sin = self.neg_freqs_sin.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq_cos, video_freq_sin = video_freq
            video_freq_cos = video_freq_cos.to(device)
            video_freq_sin = video_freq_sin.to(device)
            vid_freqs.append((video_freq_cos, video_freq_sin))

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs_cos = self.pos_freqs_cos[max_vid_index : max_vid_index + max_len, ...]
        txt_freqs_sin = self.pos_freqs_sin[max_vid_index : max_vid_index + max_len, ...]

        # Concatenate video frequencies
        vid_freqs_cos = torch.cat([vf[0] for vf in vid_freqs], dim=0)
        vid_freqs_sin = torch.cat([vf[1] for vf in vid_freqs], dim=0)

        return (vid_freqs_cos, vid_freqs_sin), (txt_freqs_cos, txt_freqs_sin)

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos_cos = self.pos_freqs_cos.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg_cos = self.neg_freqs_cos.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_pos_sin = self.pos_freqs_sin.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg_sin = self.neg_freqs_sin.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame_cos = freqs_pos_cos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        freqs_frame_sin = freqs_pos_sin[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            freqs_height_cos = torch.cat([freqs_neg_cos[1][-(height - height // 2) :], freqs_pos_cos[1][: height // 2]], dim=0)
            freqs_height_cos = freqs_height_cos.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width_cos = torch.cat([freqs_neg_cos[2][-(width - width // 2) :], freqs_pos_cos[2][: width // 2]], dim=0)
            freqs_width_cos = freqs_width_cos.view(1, 1, width, -1).expand(frame, height, width, -1)

            freqs_height_sin = torch.cat([freqs_neg_sin[1][-(height - height // 2) :], freqs_pos_sin[1][: height // 2]], dim=0)
            freqs_height_sin = freqs_height_sin.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width_sin = torch.cat([freqs_neg_sin[2][-(width - width // 2) :], freqs_pos_sin[2][: width // 2]], dim=0)
            freqs_width_sin = freqs_width_sin.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height_cos = freqs_pos_cos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width_cos = freqs_pos_cos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_height_sin = freqs_pos_sin[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width_sin = freqs_pos_sin[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs_cos = torch.cat([freqs_frame_cos, freqs_height_cos, freqs_width_cos], dim=-1).reshape(seq_lens, -1)
        freqs_sin = torch.cat([freqs_frame_sin, freqs_height_sin, freqs_width_sin], dim=-1).reshape(seq_lens, -1)

        return freqs_cos.clone().contiguous(), freqs_sin.clone().contiguous()


class GaudiQwenImagePipeline(GaudiDiffusionPipeline, QwenImagePipeline):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L132

    This class inherits from `QwenImagePipeline` and overrides methods to use Gaudi-specific implementations.
    add args use_habana
    add args use_hpu_graphs
    add args gaudi_config
    add args bf16_full_eval
    add args sdp_on_bf16
    """
    
    
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: QwenImageTransformer2DModel,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
        sdp_on_bf16: bool = False,
        is_training: bool = False,
    ):
        GaudiDiffusionPipeline.__init__(
            self,
            use_habana,
            use_hpu_graphs,
            gaudi_config,
            bf16_full_eval,
            sdp_on_bf16,
        )
        QwenImagePipeline.__init__(
            self,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        self.to(self._device)
        self.transformer.forward = types.MethodType(QwenImageTransformer2DModelGaudi, self.transformer)
        for block in self.transformer.transformer_blocks:
            block.attn.processor = GaudiQwenDoubleStreamAttnProcessor2_0(is_training)
        config = self.transformer.config
        self.transformer.pos_embed = GaudiQwenEmbedRope(theta=10000, axes_dim=list(config['axes_dims_rope']), scale_rope=True)
        
        if use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            #transformer = wrap_in_hpu_graph(transformer)
            for block in self.transformer.transformer_blocks:
                block = wrap_in_hpu_graph(block)
            

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            profiling_warmup_steps (`int`, *optional*):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*):
                Number of steps to be captured when enabling profiling.
                
        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        import habana_frameworks.torch as ht
        import habana_frameworks.torch.core as htcore
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )


        hb_profiler = HabanaProfile(
            warmup=profiling_warmup_steps,
            active=profiling_steps,
            record_shapes=False,
            name="diffuser_pipeline",
        )
        hb_profiler.start()
        
        
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        #latents, latent_image_ids = self.prepare_latents(
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
       
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
                    
                if not self.use_hpu_graphs:
                    htcore.mark_step()
                    
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()
            if not self.use_hpu_graphs:
                htcore.mark_step()
            hb_profiler.step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        hb_profiler.stop()
        
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
