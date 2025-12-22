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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Tongyi-MAI/Z-Image-Turbo",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=9,
        help="number of transformer inference steps",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="The height in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="The width in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
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

    warmup = 5
    for i in range(warmup):
        # 2. Generate Image
        pipe(
            prompt=args.prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,  # This actually results in 8 DiT forwards
            guidance_scale=args.guidance_scale,     # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
    torch.cuda.synchronize()

    inf_cnt = 5
    t0 = tm_perf.perf_counter()
    for i in range(inf_cnt):
        # 2. Generate Image
        image = pipe(
            prompt=args.prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,  # This actually results in 8 DiT forwards
            guidance_scale=args.guidance_scale,     # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]

    torch.cuda.synchronize()
    t1 = tm_perf.perf_counter()
    duration = (t1-t0)/inf_cnt
    print(f'Z-Image pipeline gaudi duration:{duration:.3f}')

    file_name = f"z_image_output_{args.width}x{args.height}.png"
    image.save(file_name)
    print(f'save {file_name} done!')

if "__main__" == __name__:
    main()
