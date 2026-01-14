# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import json
import fcntl
import random
import logging
import argparse
import warnings

import torch
import torch.distributed as dist

from wan.utils.utils import save_video, str2bool
from wan.distributed.util import init_distributed_group
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
import wan

warnings.filterwarnings('ignore')


def update_job(job_processed: list, args):
    """
    Update job status in job_animate.txt file.

    Args:
        job_processed: List of job attributes to update
        args: Command line arguments containing video_dir and sep
    """
    job_file = os.path.join(args.video_dir, "job_animate.txt")
    sep = args.sep
    if job_processed:
        with open(job_file, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                lines_before_write = [line.strip() for line in f if line.strip()]

                job_id_to_update = job_processed[0]
                found = False
                for i, line in enumerate(lines_before_write):
                    if line.startswith(job_id_to_update + sep):
                        lines_before_write[i] = sep.join(map(str, job_processed))
                        found = True
                        break

                if not found:
                    lines_before_write.append(sep.join(map(str, job_processed)))

                f.seek(0)
                f.truncate()
                for line in lines_before_write:
                    f.write(line + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def _validate_args(args):
    """Validate command line arguments."""
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.process_ckpt_dir is not None, "Please specify the process checkpoint directory."

    args.task = "animate-14B"
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Job service for Wan2.2 Animate video generation"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/hf/Wan2.2-Animate-14B",
        help="Path to Wan2.2-Animate-14B checkpoint directory."
    )
    parser.add_argument(
        "--process_ckpt_dir",
        type=str,
        default="/hf/Wan2.2-Animate-14B/process_checkpoint",
        help="Path to preprocessing model checkpoints."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Sequence parallelism size for multi-card inference."
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5."
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU."
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="dpm++",
        choices=["unipc", "dpm++"],
        help="Sampling solver algorithm."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="Diffusion sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor."
    )
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Convert DiT model parameters dtype."
    )
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora for character replacement."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/user/video",
        help="Output directory for generated videos."
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=",",
        help="Separator for job file fields."
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank: int):
    """Initialize logging based on process rank."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def run_preprocessing(args, job_dir: str, input_data: dict) -> str:
    """
    Run preprocessing pipeline for animate job.

    Args:
        args: Command line arguments
        job_dir: Directory for job files
        input_data: Input parameters from input.json

    Returns:
        Path to preprocessed data directory
    """
    # Import preprocessing modules
    # WAN_ROOT is set in Docker to the Wan2.2 installation directory
    wan_root = os.getenv("WAN_ROOT", "/home/user/Wan2.2")
    preprocess_path = os.path.join(wan_root, "wan/modules/animate/preprocess")
    sys.path.insert(0, preprocess_path)
    from process_pipepline import ProcessPipeline

    video_path = input_data["video_path"]
    image_path = input_data["image_path"]
    mode = input_data.get("mode", "animate")
    size = input_data.get("size", "1280*720")

    # Parse resolution
    width, height = map(int, size.split("*"))

    # Setup checkpoint paths
    pose2d_checkpoint_path = os.path.join(args.process_ckpt_dir, "pose2d/vitpose_h_wholebody.onnx")
    det_checkpoint_path = os.path.join(args.process_ckpt_dir, "det/yolov10m.onnx")

    replace_flag = (mode == "replace")
    sam2_checkpoint_path = os.path.join(args.process_ckpt_dir, "sam2/sam2_hiera_large.pt") if replace_flag else None

    # Create preprocessing pipeline
    process_pipeline = ProcessPipeline(
        det_checkpoint_path=det_checkpoint_path,
        pose2d_checkpoint_path=pose2d_checkpoint_path,
        sam_checkpoint_path=sam2_checkpoint_path,
        flux_kontext_path=None  # FLUX disabled by default
    )

    # Output path for preprocessed data
    preprocess_output = os.path.join(job_dir, "preprocess")
    os.makedirs(preprocess_output, exist_ok=True)

    # Run preprocessing
    process_pipeline(
        video_path=video_path,
        refer_image_path=image_path,
        output_path=preprocess_output,
        resolution_area=[width, height],
        fps=30,  # Fixed for animate-14B
        iterations=3,  # Default for replace mode
        k=7,  # Default for replace mode
        w_len=1,
        h_len=1,
        retarget_flag=(mode == "animate"),  # Enable retargeting for animate mode
        use_flux=False,
        replace_flag=replace_flag
    )

    return preprocess_output


def calculate_frame_num(seconds: int, fps: int = 30) -> int:
    """
    Calculate frame number from seconds.
    Frame number must be 4n+1 for Animate-14B.
    """
    frame_num = fps * seconds
    frame_num = ((frame_num - 1) // 4) * 4 + 1
    return frame_num


def generate(args):
    """Main generation loop."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        # For HPU/Gaudi, we don't call torch.cuda.set_device
        # The device is set via HABANA_VISIBLE_DEVICES environment variable
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1), \
            "sequence parallel is not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, \
            "The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS["animate-14B"]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, \
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Job service args: {args}")
    logging.info(f"Model config: {cfg}")
    logging.info("Creating WanAnimate pipeline.")

    wan_animate = wan.WanAnimate(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        use_relighting_lora=args.use_relighting_lora,
    )

    job_file = os.path.join(args.video_dir, "job_animate.txt")

    while True:
        try:
            time.sleep(10.0)
            if not os.path.exists(job_file):
                time.sleep(1.0)
                continue

            job_to_process = None
            if rank == 0:
                with open(job_file, "r+", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        lines = [line.strip() for line in f if line.strip()]
                        updated_lines = []
                        job_found = False

                        for line in lines:
                            parts = line.strip().split(args.sep)
                            # Job format: id,status,created_time,seconds,size,mode,fps,shift,steps,refert_num,seed,generate_duration,start_time,end_time,error_msg
                            if not job_found and len(parts) >= 15 and parts[1] in ["processing", "queued"]:
                                job_found = True
                                parts[1] = "processing"
                                parts[12] = str(int(time.time()))  # Set start time
                                job_to_process = parts
                                updated_lines.append(args.sep.join(map(str, parts)) + "\n")
                            else:
                                updated_lines.append(line + "\n")

                        if job_found:
                            f.seek(0)
                            f.truncate()
                            f.writelines(updated_lines)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)

            if world_size > 1:
                job_list = [job_to_process] if rank == 0 else [None]
                dist.broadcast_object_list(job_list, src=0)
                job_to_process = job_list[0]

            if job_to_process:
                try:
                    # Parse job info
                    # Format: id,status,created_time,seconds,size,mode,fps,shift,steps,refert_num,seed,generate_duration,start_time,end_time,error_msg
                    (job_id, status, created_str, seconds, size, mode, fps, shift, steps,
                     refert_num, seed, generate_duration, start_time, end_time, *error_msg_parts) = job_to_process

                    generate_start_time = float(start_time)
                    job_dir = os.path.join(args.video_dir, job_id)
                    os.makedirs(job_dir, exist_ok=True)

                    input_json_path = os.path.join(job_dir, "input.json")
                    video_path = os.path.join(job_dir, "output.mp4")

                    with open(input_json_path, "r", encoding="utf-8") as f:
                        input_data = json.load(f)

                    logging.info(f"Processing job {job_id}: mode={mode}, size={size}, seconds={seconds}")

                    # Step 1: Preprocessing
                    # Note: The preprocessing extracts all frames from the driving video.
                    # The output video length is determined by the driving video length,
                    # not by the 'seconds' parameter (which is used for time estimation).
                    logging.info(f"Running preprocessing for job {job_id}...")
                    preprocess_path = run_preprocessing(args, job_dir, input_data)
                    logging.info(f"Preprocessing completed: {preprocess_path}")

                    # Step 2: Generation
                    logging.info(f"Running generation for job {job_id}...")

                    # Calculate frame number
                    frame_num = calculate_frame_num(int(seconds), fps=30)

                    # Determine if replace mode
                    replace_flag = (mode == "replace")

                    # Get prompt
                    prompt = input_data.get("prompt", "视频中的人在做动作")

                    # Run generation
                    video = wan_animate.generate(
                        src_root_path=preprocess_path,
                        replace_flag=replace_flag,
                        clip_len=frame_num,
                        refert_num=int(refert_num),
                        shift=float(shift),
                        sample_solver=args.sample_solver,
                        sampling_steps=int(steps),
                        guide_scale=1,  # Fixed for animate
                        input_prompt=prompt,
                        n_prompt="",
                        seed=int(seed),
                        offload_model=args.offload_model,
                    )

                    if rank == 0:
                        logging.info(f"Saving generated video to {video_path}")
                        save_video(
                            tensor=video[None],
                            save_file=video_path,
                            fps=30,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )

                        generate_end_time = time.time()
                        job_processed = [
                            job_id, "completed", created_str, seconds, size, mode, fps,
                            shift, steps, refert_num, seed,
                            max(0, int(generate_end_time - generate_start_time)),
                            int(generate_start_time), int(generate_end_time), ""
                        ]
                        update_job(job_processed, args)
                        logging.info(f"Job {job_id} completed successfully.")

                except Exception as e:
                    import traceback
                    logging.error(f"Error processing job {job_id}: {e}\n{traceback.format_exc()}")
                    if rank == 0:
                        generate_end_time = time.time()
                        job_processed = [
                            job_id, "error", created_str, seconds, size, mode, fps,
                            shift, steps, refert_num, seed,
                            max(0, int(generate_end_time - generate_start_time)),
                            int(generate_start_time), int(generate_end_time), str(e)
                        ]
                        update_job(job_processed, args)

        except Exception as e:
            import traceback
            logging.error(f"Job worker encountered an error: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
