import os
import torch
import torch.nn.functional as F
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, PreTrainedModel
from diffusers.image_processor import PipelineImageInput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from diffusers.models.controlnets import ZImageControlNetModel
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
from diffusers.pipelines.z_image.pipeline_z_image_controlnet_inpaint import calculate_shift, retrieve_timesteps, retrieve_latents
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers import transformer_z_image
from diffusers.models.controlnets import controlnet_z_image
from diffusers import ZImageControlNetInpaintPipeline

from optimum.utils import logging
from optimum.habana.diffusers.pipelines.pipeline_utils import GaudiDiffusionPipeline
from optimum.habana.diffusers.models.attention_processor import FlashAttnV3Gaudi
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.diffusers.models.unet_2d_condition import set_default_attn_processor_hpu
from .pipeline_zimage_controlnet import _Zimage_tranformer_prepare_sequence_gaudi, _Zimage_transformer_build_unified_sequence_gaudi, controlnet_forward_gaudi, Zimage_transformer_forward_gaudi 

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingMode, apply_rotary_pos_emb

logger = logging.get_logger(__name__)

class ZSingleStreamAttnProcessorGaudi:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        self.use_habana=True
        self.fav3 = FlashAttnV3Gaudi()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor,
                             freqs_cis: torch.Tensor,
                             use_real: bool = True,
                             use_real_unbind_dim:int = -1,
        )-> torch.Tensor:
            if use_real:
                freqs_cis = freqs_cis.unsqueeze(2)
                cos, sin=  freqs_cis.unbind(-1)
                cos = torch.repeat_interleave(cos, 2, dim=-1)
                sin = torch.repeat_interleave(sin, 2, dim=-1)
                out = apply_rotary_pos_emb(x_in, cos, sin, None, 0, RotaryPosEmbeddingMode.PAIRWISE)

                return out.type_as(x_in)
            else:
                with torch.amp.autocast("hpu", enabled=False):
                    x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                    freqs_cis = freqs_cis.unsqueeze(2)
                    x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                    return x_out.type_as(x_in)  # todo

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        fsdpa_mode = 'fp32' if os.environ.get('FP32_SOFTMAX_VISION', 'false').lower() in ['true', '1' ] else 'fast'
        hidden_states = self.fav3.forward(query, key, value, attention_mask, fsdpa_mode)

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output

class RopeEmbedderGaudi:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("hpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                cos = torch.cos(freqs).unsqueeze(-1)
                sin = torch.sin(freqs).unsqueeze(-1)
                freqs_cis_i = rearrange([cos, sin], ' b s n h -> s n (h b)')
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])

        return torch.cat(result, dim=-2)

setattr(transformer_z_image, "RopeEmbedder", RopeEmbedderGaudi)
setattr(controlnet_z_image, "RopeEmbedder", RopeEmbedderGaudi)

class GaudiZImageControlNetInpaintPipeline(GaudiDiffusionPipeline, ZImageControlNetInpaintPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
        controlnet: ZImageControlNetModel,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
        sdp_on_bf16: bool = False,
    ):
        GaudiDiffusionPipeline.__init__(
            self,
            use_habana,
            use_hpu_graphs,
            gaudi_config,
            bf16_full_eval,
            sdp_on_bf16,
        )
        ZImageControlNetInpaintPipeline.__init__(
            self,
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            transformer,
            controlnet
        )
        n_refiner_layers =len(self.transformer.noise_refiner)
        if self.use_hpu_graphs:
            for i in range(n_refiner_layers):
                self.transformer.noise_refiner[i] = ht.hpu.wrap_in_hpu_graph(self.transformer.noise_refiner[i])

            for i in range(n_refiner_layers):
                self.transformer.context_refiner[i] = ht.hpu.wrap_in_hpu_graph(self.transformer.context_refiner[i])

            for i in range(len(self.transformer.layers)):
                self.transformer.layers[i] = ht.hpu.wrap_in_hpu_graph(self.transformer.layers[i])

        self.transformer.forward = types.MethodType(Zimage_transformer_forward_gaudi, self.transformer)
        self.controlnet.forward = types.MethodType(controlnet_forward_gaudi, self.controlnet)

        self.transformer._prepare_sequence = types.MethodType(_Zimage_tranformer_prepare_sequence_gaudi, self.transformer)
        self.transformer._build_unified_sequence = types.MethodType(_Zimage_transformer_build_unified_sequence_gaudi, self.transformer)

        for layer in self.transformer.noise_refiner:
            layer.attention.set_processor(ZSingleStreamAttnProcessorGaudi())

        for layer in self.transformer.context_refiner:
            layer.attention.set_processor(ZSingleStreamAttnProcessorGaudi())
        
        if not self.transformer.siglip_refiner is None:
            for layer in self.transformer.siglip_refiner:
                layer.attention.set_processor(ZSingleStreamAttnProcessorGaudi())

        for layer in self.transformer.layers:
            layer.attention.set_processor(ZSingleStreamAttnProcessorGaudi())

        for layer in self.controlnet.control_layers:
            layer.attention.set_processor(ZSingleStreamAttnProcessorGaudi())

        for layer in self.controlnet.control_noise_refiner:
            layer.attention.set_processor(ZSingleStreamAttnProcessorGaudi())

        self.to(self._device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.75,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_normalization (`bool`, *optional*, defaults to False):
                Whether to apply configuration normalization.
            cfg_truncation (`float`, *optional*, defaults to 1.0):
                The truncation value for configuration.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.ZImagePipelineOutput`] instead of a plain
                tuple.
            joint_attention_kwargs (`dict`, *optional*):
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
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.z_image.ZImagePipelineOutput`] or `tuple`: [`~pipelines.z_image.ZImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            height = (height // vae_scale + 1) * vae_scale
            logger.warning(f'pad height to {height}')
        if width % vae_scale != 0:
            width = (width // vae_scale + 1) * vae_scale
            logger.warning(f'pad width to {width}')

        if height > 2048:
            height = 2048
            logger.warning(f'resize height to {height}')
        if width > 2048:
            width = 2048
            logger.warning(f'resize width to {width}')

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        control_image = self.prepare_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.vae.dtype,
        )
        height, width = control_image.shape[-2:]
        control_image = retrieve_latents(self.vae.encode(control_image), generator=generator, sample_mode="argmax")
        control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        control_image = control_image.unsqueeze(2)

        mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width)
        mask_condition = torch.tile(mask_condition, [1, 3, 1, 1]).to(
            device=control_image.device, dtype=control_image.dtype
        )

        init_image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.vae.dtype,
        )
        height, width = init_image.shape[-2:]
        init_image = init_image * (mask_condition < 0.5)
        init_image = retrieve_latents(self.vae.encode(init_image), generator=generator, sample_mode="argmax")
        init_image = (init_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        init_image = init_image.unsqueeze(2)

        mask_condition = F.interpolate(1 - mask_condition[:, :1], size=init_image.size()[-2:], mode="nearest").to(
            device=control_image.device, dtype=control_image.dtype
        )
        mask_condition = mask_condition.unsqueeze(2)

        control_image = torch.cat([control_image, mask_condition, init_image], dim=1)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # 5. Prepare timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps)):
                if self.interrupt:
                    continue

                t = timesteps[0]
                timesteps = torch.roll(timesteps, shifts=-1, dims=0)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                # Normalized time for time-aware config (0 at start, 1 at end)
                t_norm = timestep[0].item()

                # Handle cfg truncation
                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                # Run CFG only if configured AND scale is non-zero
                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                if apply_cfg:
                    latents_typed = latents.to(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = timestep.repeat(2)
                else:
                    latent_model_input = latents.to(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                controlnet_block_samples = self.controlnet(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    control_image,
                    conditioning_scale=controlnet_conditioning_scale,
                )
                htcore.mark_step()

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    controlnet_block_samples=controlnet_block_samples,
                )[0]
                htcore.mark_step()

                if apply_cfg:
                    # Perform CFG
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        # Renormalization
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                htcore.mark_step()
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
                assert latents.dtype == torch.float32

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        htcore.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ZImagePipelineOutput(images=image)
