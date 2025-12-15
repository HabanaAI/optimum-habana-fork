import torch
import argparse
import random
import numpy as np
import time as tm_perf

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.gpu_migration

from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.diffusers import GaudiStableDiffusionZImagePipeline

def set_seed():
    seed = 5451
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        nargs="*",
        default="An image of a squirrel in Picasso style",
        help="The prompt or prompts to guide the image generation.",
    )
    args = parser.parse_args()

    set_seed()
    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = False
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": gaudi_config,
    }

    model_name_path = args.model_name_or_path
    # 1. Load the pipeline
    # Use bfloat16 for optimal performance on supported GPUs
    pipe = GaudiStableDiffusionZImagePipeline.from_pretrained(
        model_name_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        **kwargs,
    )
    pipe.to("hpu")
    
    prompt = args.prompts
    warmup = 5
    width = 512
    height = 512

    for i in range(warmup):
        # 2. Generate Image
        pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=9,  # This actually results in 8 DiT forwards
            guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cpu").manual_seed(42),
        ).images[0]
    torch.cuda.synchronize()

    inf_cnt = 5
    t0 = tm_perf.perf_counter()
    for i in range(inf_cnt):
        # 2. Generate Image
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=9,  # This actually results in 8 DiT forwards
            guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cpu").manual_seed(42),
        ).images[0]

    torch.cuda.synchronize()
    t1 = tm_perf.perf_counter()
    duration = (t1-t0)/inf_cnt
    print(f'Z-Image pipeline gaudi duration:{duration:.3f}')
    
    image.save("example.png")

if "__main__" == __name__:
    main()
