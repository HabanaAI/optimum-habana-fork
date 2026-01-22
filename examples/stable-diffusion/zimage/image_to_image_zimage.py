import torch
import argparse
import random
import numpy as np
import time

import habana_frameworks.torch as ht
from diffusers import ZImageControlNetModel
from diffusers.utils import load_image

from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.diffusers import GaudiZImageControlNetPipeline, GaudiZImageControlNetInpaintPipeline


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
        "--controlnet_path",
        default="Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--image_path",
        default="inpaint.jpg",
        type=str,
        help="Z-image controlnet inpaint image path",
    )
    parser.add_argument(
        "--pipeline_type",
        default="controlnet",
        type=str,
        help="Z-image controlnet pipeline type",
    )
    parser.add_argument(
        "--mask_path",
        default="mask.jpg",
        type=str,
        help="Z-image controlnet inpaint mask image path",
    )
    parser.add_argument(
        "--pose_path",
        default="pose.jpg",
        type=str,
        help="Z-image controlnet inpaint pose image path",
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
        default=1024,
        help="The height in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=0.75,
        help="A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        default=False,
        help="Use HPU graphs to accelerate inference. Suggest not to enable it for large figure generation",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Number of benchmark loops for generation.",
    )
    args = parser.parse_args()
    print(f'get args:\n{args} \n\n')
    set_seed(args.seed)

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = False
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }
    model_name_path = args.model_name_or_path

    #config='hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.1/config.json',
    controlnet = ZImageControlNetModel.from_single_file(
            args.controlnet_path,
            config='config.json',
            torch_dtype = torch.bfloat16
    )

    # 1. Load the pipeline
    # Use bfloat16 for optimal performance on supported GPUs
    if args.pipeline_type == 'controlnet':
        control_image = load_image(args.pose_path)
        pipe = GaudiZImageControlNetPipeline.from_pretrained(
            model_name_path,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
    elif args.pipeline_type == 'controlnet_inpaint':
        image = load_image(args.image_path)
        mask_image = load_image(args.mask_path)
        control_image = load_image(args.pose_path)
        pipe = GaudiZImageControlNetInpaintPipeline.from_pretrained(
                model_name_path, 
                controlnet = controlnet,
                torch_dtype=torch.bfloat16, 
                **kwargs
        )
    else:
        print(f'ERROR unsuppported pipeline type:{args.pipeline_type}')
        exit()
    pipe.to("hpu")

    print('load image and pipeline done, inference...')
    
    for i in range(args.loop):
        t0 = time.time()
        # 2. Generate Image
        if args.pipeline_type == 'controlnet':
            image_out = pipe(
                prompt=args.prompts,
                control_image=control_image,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0]
        elif args.pipeline_type == 'controlnet_inpaint':
            image_out = pipe(
                prompt=args.prompts,
                image=image,
                mask_image=mask_image,
                control_image=control_image,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0]
        torch.hpu.synchronize()
        t1 = time.time()
        duration = t1 - t0
        print(f"Z-Image {args.pipeline_type} Pipeline Latency in Loop #{i:d}: {duration:.1f} sec")
    file_name = f"z_image_{args.pipeline_type}_{args.width}x{args.height}.png"
    image_out.save(file_name)
    print(f'Completed saving {file_name}!')

if "__main__" == __name__:
    main()
