import torch
import time
import os, sys
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.gpu_migration
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.diffusers import GaudiStableDiffusionXLKolorsInpaintPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Kolors",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="An image of a squirrel in Picasso style",
        help="The prompt or prompts to guide the image generation.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="Path to pre-trained model",
    )
    args = parser.parse_args()

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = False
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": gaudi_config,
    }

    ckpt_dir = args.model_name_or_path
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    pipe = GaudiStableDiffusionXLKolorsInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            **kwargs,
    )
    
    pipe.to("hpu")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(603)
    basename = args.image_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    image = Image.open(args.image_path).convert('RGB')
    mask_image = Image.open(args.mask_path).convert('RGB')

    warmup = 3
    for i in range(warmup):
        result = pipe(
            prompt = args.prompts,
            image = image,
            mask_image = mask_image,
            height=1024,
            width=768,
            guidance_scale = 6.0,
            generator= generator,
            num_inference_steps= 5,
            negative_prompt = '残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量',
            num_images_per_prompt = 1,
            strength = 0.999
        ).images[0]
    torch.hpu.synchronize()

    iter_time = 0
    start_time = time.perf_counter()
    for i in range(5):
        result = pipe(
            prompt = args.prompts,
            image = image,
            mask_image = mask_image,
            height=1024,
            width=768,
            guidance_scale = 6.0,
            generator= generator,
            num_inference_steps= 25,
            negative_prompt = '残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量',
            num_images_per_prompt = 1,
            strength = 0.999
        ).images[0]

    torch.hpu.synchronize()
    iter_time += time.perf_counter() - start_time

    iter_time = iter_time / 5
    print(f'Kolors in-Painting pipeline duration:{iter_time:.3f}s')

    result.save(f'sample_inpainting_3.jpg')

if __name__ == '__main__':
    main()
