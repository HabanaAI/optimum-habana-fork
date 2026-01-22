# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import numpy as np
import PIL.Image
import torch
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, compute_empirical_mu, retrieve_timesteps
from diffusers.utils import BaseOutput, replace_example_docstring
from optimum.utils import logging
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ...models.attention_processor import GaudiFlux2AttnProcessor, GaudiFlux2ParallelSelfAttnProcessor
from ...schedulers import GaudiFlowMatchEulerDiscreteScheduler
from ..pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class GaudiFluxPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    throughput: float


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from optimum.habana.diffusers import GaudiFlux2Pipeline
        >>> from transformers import PixtralProcessor, Mistral3ForConditionalGeneration

        >>> model_id = "black-forest-labs/FLUX.2-dev"
        >>> pipe = GaudiFlux2Pipeline.from_pretrained(
        ...     model_id,
        ...     torch_dtype=torch.bfloat16,
        ...     use_habana=True,
        ...     use_hpu_graphs=True,
        ...     gaudi_config="Habana/stable-diffusion",
        ...     text_encoder=None,
        ... )
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.

        >>> tokenizer = PixtralProcessor.from_pretrained(model_id, subfolder="tokenizer")
        >>> text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        ...     model_id, subfolder="text_encoder", torch_dtype=torch.float32, attn_implementation="eager").to("cpu")
        >>> prompt_embeds_cpu = GaudiFlux2Pipeline._get_mistral_3_small_prompt_embeds(
        ...     text_encoder=text_encoder, tokenizer=tokenizer, prompt=prompt, device=torch.device("cpu"), dtype=torch.float32)
        >>> prompt_embeds_hpu = prompt_embeds_cpu.to("hpu", dtype=torch.bfloat16)

        >>> image = pipe(prompt_embeds=prompt_embeds_hpu, num_inference_steps=50, guidance_scale=2.5).images[0]
        >>> image.save("flux2.png")
        ```
"""


class GaudiFlux2Pipeline(GaudiDiffusionPipeline, Flux2Pipeline):
    r"""
    Adapted from https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/flux2/pipeline_flux2.py#L251

    The Flux2 pipeline for text-to-image generation.

    Reference: [https://bfl.ai/blog/flux-2](https://bfl.ai/blog/flux-2)

    Args:
        transformer ([`Flux2Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLFlux2`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Mistral3ForConditionalGeneration`]):
            [Mistral3ForConditionalGeneration](https://huggingface.co/docs/transformers/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration)
        tokenizer (`AutoProcessor`):
            Tokenizer of class
            [PixtralProcessor](https://huggingface.co/docs/transformers/en/model_doc/pixtral#transformers.PixtralProcessor).
        use_habana (bool, defaults to `False`):
            Whether to use Gaudi (`True`) or CPU (`False`).
        use_hpu_graphs (bool, defaults to `False`):
            Whether to use HPU graphs or not.
        gaudi_config (Union[str, [`GaudiConfig`]], defaults to `None`):
            Gaudi configuration to use. Can be a string to download it from the Hub.
            Or a previously initialized config can be passed.
        bf16_full_eval (bool, defaults to `False`):
            Whether to use full bfloat16 evaluation instead of 32-bit.
            This will be faster and save memory compared to fp32/mixed precision but can harm generated images.
        sdp_on_bf16 (bool, defaults to `False`):
            Whether to allow PyTorch to use reduced precision in the SDPA math backend.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: GaudiFlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        transformer: Flux2Transformer2DModel,
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
        Flux2Pipeline.__init__(
            self,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )

        for block in self.transformer.single_transformer_blocks:
            block.attn.processor = GaudiFlux2ParallelSelfAttnProcessor()
        for block in self.transformer.transformer_blocks:
            block.attn.processor = GaudiFlux2AttnProcessor()

        self.to(self._device)
        if use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            self.transformer = wrap_in_hpu_graph(self.transformer)

    @classmethod
    def _split_inputs_into_batches(cls, batch_size, latents, image_latents, prompt_embeds, guidance):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        if image_latents is not None:
            image_latents_batches = list(torch.split(image_latents, batch_size))
        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))
        if guidance is not None:
            guidance_batches = list(torch.split(guidance, batch_size))

        # If the last batch has less samples than batch_size, pad it with dummy samples
        num_dummy_samples = 0
        if latents_batches[-1].shape[0] < batch_size:
            num_dummy_samples = batch_size - latents_batches[-1].shape[0]

            # Pad latents_batches
            sequence_to_stack = (latents_batches[-1],) + tuple(
                torch.zeros_like(latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            latents_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad image_latents_batches if necessary
            if image_latents is not None:
                sequence_to_stack = (image_latents_batches[-1],) + tuple(
                    torch.zeros_like(image_latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                image_latents_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad prompt_embeds_batches
            sequence_to_stack = (prompt_embeds_batches[-1],) + tuple(
                torch.zeros_like(prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad guidance if necessary
            if guidance is not None:
                guidance_batches[-1] = guidance_batches[-1].unsqueeze(1)
                sequence_to_stack = (guidance_batches[-1],) + tuple(
                    torch.zeros_like(guidance_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                guidance_batches[-1] = torch.vstack(sequence_to_stack).squeeze(1)

        # Stack batches in the same tensor
        latents_batches = torch.stack(latents_batches)
        image_latents_batches = torch.stack(image_latents_batches) if image_latents is not None else None
        prompt_embeds_batches = torch.stack(prompt_embeds_batches)
        guidance_batches = torch.stack(guidance_batches) if guidance is not None else None

        return (
            latents_batches,
            image_latents_batches,
            prompt_embeds_batches,
            guidance_batches,
            num_dummy_samples,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        caption_upsample_temperature: float = None,
        timesteps: List[int] = None,
        batch_size: int = 1,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        r"""
        Adapted from https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/flux2/pipeline_flux2.py#L745
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
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
            text_encoder_out_layers (`Tuple[int]`):
                Layer indices to use in the `text_encoder` to derive the final prompt embeddings.
            caption_upsample_temperature (`float`):
                When specified, we will try to perform caption upsampling for potentially improved outputs. We
                recommend setting it to 0.15 if caption upsampling is to be performed.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            profiling_warmup_steps (`int`, *optional*):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*):
                Number of steps to be captured when enabling profiling.

        Examples:

        Returns:
            [`~pipelines.flux2.Flux2PipelineOutput`] or `tuple`: [`~pipelines.flux2.Flux2PipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            num_prompts = 1
        elif prompt is not None and isinstance(prompt, list):
            num_prompts = len(prompt)
        else:
            num_prompts = prompt_embeds.shape[0]
        num_batches = math.ceil((num_images_per_prompt * num_prompts) / batch_size)

        device = self._execution_device

        # 3. prepare text embeddings
        if caption_upsample_temperature:
            prompt = self.upsample_prompt(
                prompt, images=image, temperature=caption_upsample_temperature, device=device
            )
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=num_prompts * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=num_prompts * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.bfloat16)
        guidance = guidance.expand(latents.shape[0])

        logger.info(
            f"{num_prompts} prompt(s) received, {num_images_per_prompt} generation(s) per prompt,"
            f" {batch_size} sample(s) per batch, {num_batches} total batch(es)."
        )
        if num_batches < 3:
            logger.warning("The first two iterations are slower so it is recommended to feed more batches.")

        throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
        use_warmup_inference_steps = (
            num_batches <= throughput_warmup_steps and num_inference_steps > throughput_warmup_steps
        )

        ht.hpu.synchronize()
        t0 = time.time()
        t1 = t0

        hb_profiler = HabanaProfile(
            warmup=profiling_warmup_steps,
            active=profiling_steps,
            record_shapes=False,
            name="diffuser_pipeline",
        )
        hb_profiler.start()

        # 5.1. Split Input data to batches (HPU-specific step)
        (
            latents_batches,
            image_latents_batches,
            prompt_embeds_batches,
            guidance_batches,
            num_dummy_samples,
        ) = self._split_inputs_into_batches(batch_size, latents, image_latents, prompt_embeds, guidance)

        outputs = {
            "images": [],
        }

        # 6. Denoising loop
        for j in range(num_batches):
            # The throughput is calculated from the 4th iteration
            # because compilation occurs in the first 2-3 iterations
            if j == throughput_warmup_steps:
                ht.hpu.synchronize()
                t1 = time.time()

            latents_batch = latents_batches[0].clone()  # avoid shallow copy
            latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
            image_latents_batch = None if image_latents_batches is None else image_latents_batches[0]
            image_latents_batches = None if image_latents_batches is None else torch.roll(image_latents_batches, shifts=-1, dims=0)
            prompt_embeds_batch = prompt_embeds_batches[0]
            prompt_embeds_batches = torch.roll(prompt_embeds_batches, shifts=-1, dims=0)
            guidance_batch = None if guidance_batches is None else guidance_batches[0]
            guidance_batches = None if guidance_batches is None else torch.roll(guidance_batches, shifts=-1, dims=0)

            if hasattr(self.scheduler, "_init_step_index"):
                # Reset scheduler step index for next batch
                self.scheduler.timesteps = timesteps
                self.scheduler._init_step_index(timesteps[0])

            for i in self.progress_bar(range(len(timesteps))):
                if use_warmup_inference_steps and i == throughput_warmup_steps and j == num_batches - 1:
                    ht.hpu.synchronize()
                    t1 = time.time()

                if self.interrupt:
                    continue

                timestep = timesteps[0]
                timesteps = torch.roll(timesteps, shifts=-1, dims=0)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = timestep.expand(latents_batch.shape[0]).to(latents_batch.dtype)

                latent_model_input = latents_batch.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents_batch is not None:
                    latent_model_input = torch.cat([latents_batch, image_latents_batch], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,  # (B, image_seq_len, C)
                    timestep=timestep / 1000,
                    guidance=guidance_batch,
                    encoder_hidden_states=prompt_embeds_batch,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=latent_image_ids,  # B, image_seq_len, 4
                    joint_attention_kwargs=self._attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents_batch.size(1) :]

                # compute the previous noisy sample x_t -> x_t-1
                latents_batch = self.scheduler.step(noise_pred, timestep, latents_batch, return_dict=False)[0]

                hb_profiler.step()
                # htcore.mark_step(sync=True)
                if num_batches > throughput_warmup_steps:
                    ht.hpu.synchronize()

            if output_type == "latent":
                image = latents_batch
            else:
                latents_batch = self._unpack_latents_with_ids(latents_batch, latent_ids)

                latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents_batch.device, latents_batch.dtype)
                latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                    latents_batch.device, latents_batch.dtype
                )
                latents_batch = latents_batch * latents_bn_std + latents_bn_mean
                latents_batch = self._unpatchify_latents(latents_batch)

                image = self.vae.decode(latents_batch, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)

            outputs["images"].append(image)
            # htcore.mark_step(sync=True)

        # 7. Stage after denoising
        hb_profiler.stop()

        ht.hpu.synchronize()
        speed_metrics_prefix = "generation"
        if use_warmup_inference_steps:
            t1 = warmup_inference_steps_time_adjustment(t1, t1, num_inference_steps, throughput_warmup_steps)
        speed_measures = speed_metrics(
            split=speed_metrics_prefix,
            start_time=t0,
            num_samples=batch_size
            if t1 == t0 or use_warmup_inference_steps
            else (num_batches - throughput_warmup_steps) * batch_size,
            num_steps=batch_size * num_inference_steps
            if use_warmup_inference_steps
            else (num_batches - throughput_warmup_steps) * batch_size * num_inference_steps,
            start_time_after_warmup=t1,
        )
        logger.info(f"Speed metrics: {speed_measures}")

        # 8. Output Images
        if num_dummy_samples > 0:
            # Remove dummy generations if needed
            outputs["images"][-1] = outputs["images"][-1][:-num_dummy_samples]

        # Process generated images
        for i, image in enumerate(outputs["images"][:]):
            if i == 0:
                outputs["images"].clear()

            if output_type == "pil" and isinstance(image, list):
                outputs["images"] += image
            elif output_type in ["np", "numpy"] and isinstance(image, np.ndarray):
                if len(outputs["images"]) == 0:
                    outputs["images"] = image
                else:
                    outputs["images"] = np.concatenate((outputs["images"], image), axis=0)
            else:
                if len(outputs["images"]) == 0:
                    outputs["images"] = image
                else:
                    outputs["images"] = torch.cat((outputs["images"], image), 0)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return outputs["images"]

        return GaudiFluxPipelineOutput(
            images=outputs["images"],
            throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
        )
