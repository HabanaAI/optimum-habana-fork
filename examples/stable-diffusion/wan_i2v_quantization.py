#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

# Adapted from ../stable-diffusion/text_to_image_generation.py

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np

import torch
from diffusers.utils.export_utils import export_to_video
from diffusers.utils import load_image

from optimum.habana.diffusers import GaudiCogVideoXPipeline, GaudiTextToVideoSDPipeline, GaudiWanPipeline, GaudiWanImageToVideoPipeline
from optimum.habana.distributed import parallel_state
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.utils import set_seed
import pandas as pd


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.19.0.dev0")


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model_name_or_path",
        default="ali-vilab/text-to-video-ms-1.7b",
        type=str,
        help="Path to pre-trained model",
    )
    # Pipeline arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Spiderman is surfing",
        help="The prompt to guide the video generation.",
    )
    parser.add_argument(
        "--negative_prompts",
        type=str,
        nargs="*",
        default="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
        help="The prompt or prompts not to guide the image generation.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="The image to guide the video generation.",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="The number of videos to generate per prompt."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The number of videos in a batch.")
    parser.add_argument(
        "--max_area",
        type=int,
        default=0,
        help="The height in pixels of the generated videos (0=default from model config).",
    )
    parser.add_argument("--num_frames", type=int, default=20, help="The number of frames in the generated videos.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality videos at the expense"
            " of slower inference."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help=(
            "Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)."
            " Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,"
            " usually at the expense of lower video quality."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help=(
            "Frames per second. The rate at which the generated images shall be exported to a video after generation."
            " Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["mp4", "np"],
        default="mp4",
        help="Whether to return mp4 or Numpy arrays.",
    )

    parser.add_argument(
        "--pipeline_save_dir",
        type=str,
        default=None,
        help="The directory where the generation pipeline will be saved.",
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        default="./generated-videos",
        help="The directory where videos will be saved.",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Determines how many ranks are divided into context parallel group.",
    )
    parser.add_argument(
        "--quant_mode",
        default="disable",
        choices=["measure", "quantize", "quantize-mixed", "disable"],
        type=str,
        help="Quantization mode 'measure', 'quantize', 'quantize-mixed' or 'disable'",
    )
    parser.add_argument(
        "--quant_config",
        type=str,
        default=None,
        help="Quantization config for transformer.",
    )
    parser.add_argument(
        "--quant_config_2",
        type=str,
        default=None,
        help="Quantization config for transformer_2.",
    )
    parser.add_argument(
        "--num_calib_sample",
        type=int,
        default=2,
        help="Number of sample data used for the calibration.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    # HPU-specific arguments
    parser.add_argument("--use_habana", action="store_true", help="Use HPU.")
    parser.add_argument(
        "--use_hpu_graphs", action="store_true", help="Use HPU graphs on HPU. This should lead to faster generations."
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp32", "autocast_bf16"],
        help="Which runtime dtype to perform generation in.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="cat_wan_video",
        help="The directory where videos will be saved.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="WenhaoWang/TIP-I2V",
        help="The directory where videos will be saved.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Number of benchmark loops for generation.",
    )


    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Arguments: {args}")

    # Set video resolution
    kwargs_call = {}
    kwargs_call["num_frames"] = args.num_frames

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    if args.dtype == "autocast_bf16":
        gaudi_config_kwargs["use_torch_autocast"] = True

    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    logger.info(f"Gaudi Config: {gaudi_config}")

    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }
    if args.dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "fp32":
        kwargs["torch_dtype"] = torch.float32

    kwargs_call["quant_mode"] = args.quant_mode

    if args.context_parallel_size > 1 and parallel_state.is_unitialized():
        if not torch.distributed.is_initialized():
            import deepspeed

            torch.distributed.init_process_group(backend="hccl")
            deepspeed.init_distributed(dist_backend="hccl")
        parallel_state.initialize_model_parallel(sequence_parallel_size=args.context_parallel_size, use_fp8=False)

    # Generate images
    pipeline: GaudiWanPipeline = GaudiWanImageToVideoPipeline.from_pretrained(args.model_name_or_path, **kwargs)

    if args.quant_mode == "measure" or args.quant_mode == "quantize":
        from neural_compressor.torch.quantization import FP8Config, convert, prepare
        if pipeline.transformer is not None:
            config = FP8Config.from_json_file(args.quant_config)
            if config.measure:
                pipeline.transformer = prepare(pipeline.transformer, config)
            elif config.quantize:
                pipeline.transformer = convert(pipeline.transformer, config)

    set_seed(args.seed)
    max_area = args.max_area

    generator= torch.Generator("cpu")
    generator.manual_seed(args.seed)

    if args.quant_mode == "measure":
        import pandas as pd
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split='Eval', streaming=True)
        df = pd.DataFrame(ds)
        for i in range(args.num_calib_sample):
            prompt = df["Text_Prompt"][i]
            image = df["Image_Prompt"][i]

            aspect_ratio = image.height / image.width
            mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            image = image.resize((width, height))

            outputs = pipeline(
                    image=image,
                    prompt=prompt,
                    num_videos_per_prompt=args.num_videos_per_prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    guidance_scale=5.0,
                    negative_prompt=args.negative_prompts,
                    generator=generator,
                )
    else:
        image = load_image(args.image_path)
        aspect_ratio = image.height / image.width
        mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        for i in range(args.loop):
            t0 = time.time()
            outputs = pipeline(
                    image=image,
                    prompt=args.prompt,
                    num_videos_per_prompt=args.num_videos_per_prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    guidance_scale=5.0,
                    negative_prompt=args.negative_prompts,
                    generator=generator,
               )
            torch.hpu.synchronize()
            t1 = time.time()
            duration = t1 - t0

            if (args.context_parallel_size > 1 and torch.distributed.get_rank() == 0) or args.context_parallel_size == 1:
                logger.info("Wan Pipeline FP8 Latency in loop #{:d}: {:.1f} sec".format(i, duration))

    if args.quant_mode == "measure":
        from neural_compressor.torch.quantization import finalize_calibration
        if pipeline.transformer is not None:
            finalize_calibration(pipeline.transformer)

    # Save the pipeline in the specified directory if not None
    if args.pipeline_save_dir is not None:
        pipeline.save_pretrained(args.pipeline_save_dir)

    # Save videos in the specified directory if not None
    if args.video_save_dir is not None:
        if args.output_type == "mp4":
            video_save_dir = Path(args.video_save_dir)
            video_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving videos in {video_save_dir.resolve()}...")

            for i, video in enumerate(outputs.frames):
                filename = video_save_dir / f"{args.filename}_{i + 1}.mp4"
                export_to_video(video, str(filename.resolve()), fps=args.fps)
        else:
            logger.warning("--output_type should be equal to 'mp4' to save videos in --video_save_dir.")


if __name__ == "__main__":
    main()
