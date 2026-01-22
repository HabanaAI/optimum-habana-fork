# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Generation service for Wan2.2 Animate video generation.

This service runs on HPU with multi-card support and handles the generation phase:
- Picks up jobs with status 'preprocessed'
- Marks them as 'processing'
- Reads preprocessing metadata from preprocess_info.json
- Runs video generation on HPU
- Marks jobs as 'completed' (or 'error' if fails)

Usage:
    torchrun --nproc-per-node 8 job_service_generate.py --ckpt_dir /path/to/model --video_dir /path/to/videos
"""

import os
import sys
import time
import json
import fcntl
import logging
import argparse
import warnings
import base64
import traceback
import subprocess

import torch
import torch.distributed as dist

from wan.utils.utils import save_video, str2bool
from wan.distributed.util import init_distributed_group
from wan.configs import WAN_CONFIGS
import wan

warnings.filterwarnings('ignore')


def encode_error_msg(error_msg: str) -> str:
    """
    Encode error message to base64 to handle special characters (newlines, commas).

    Args:
        error_msg: Raw error message string

    Returns:
        Base64 encoded string, or empty string if input is empty
    """
    if not error_msg:
        return ""
    return base64.b64encode(error_msg.encode('utf-8')).decode('ascii')


def merge_audio_to_video(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    Merge audio track into video file using ffmpeg.

    Args:
        video_path: Path to the input video (without audio)
        audio_path: Path to the audio file
        output_path: Path to save the merged video with audio

    Returns:
        bool: True if merge was successful, False otherwise
    """
    try:
        # Use ffmpeg to merge audio and video
        # -shortest ensures the output matches the shortest stream (video or audio)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and os.path.exists(output_path):
            logging.info(f"Audio merged successfully to {output_path}")
            return True
        else:
            logging.warning(f"Failed to merge audio: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logging.warning("Audio merging timed out.")
        return False
    except Exception as e:
        logging.warning(f"Failed to merge audio: {e}")
        return False


def update_job(job_processed: list, args):
    """
    Update job status in job_animate.txt file using atomic write.

    Uses write-to-temp-then-rename pattern for crash safety.
    Job format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded

    Args:
        job_processed: List of job attributes to update
        args: Command line arguments containing video_dir and sep
    """
    job_file = os.path.join(args.video_dir, "job_animate.txt")
    temp_file = job_file + ".tmp"
    lock_file = job_file + ".lock"
    sep = args.sep

    if job_processed:
        # Use a separate lock file for atomic operations
        # Use "a" mode to avoid truncating the lock file which can cause race conditions
        with open(lock_file, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                # Read existing content
                lines_before_write = []
                if os.path.exists(job_file):
                    with open(job_file, "r", encoding="utf-8") as f:
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

                # Write to temp file first
                with open(temp_file, "w", encoding="utf-8") as f:
                    for line in lines_before_write:
                        f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename (this is atomic on POSIX systems)
                os.replace(temp_file, job_file)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


def _validate_args(args):
    """Validate command line arguments."""
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."

    args.task = "animate-14B"
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generation service for Wan2.2 Animate video generation"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/hf/Wan2.2-Animate-14B",
        help="Path to Wan2.2-Animate-14B checkpoint directory."
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
        default="unipc",
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
        default=True,
        help="Convert DiT model parameters dtype."
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=True,
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
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds."
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


def load_preprocess_info(job_dir: str) -> dict:
    """
    Load preprocessing metadata from preprocess_info.json.

    Args:
        job_dir: Directory for job files

    Returns:
        dict: Preprocessing metadata
    """
    preprocess_info_path = os.path.join(job_dir, "preprocess_info.json")
    with open(preprocess_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_generation_service(args):
    """Main generation service loop."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        # For HPU/Gaudi, use hccl backend
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

    logging.info(f"Generation service args: {args}")
    logging.info(f"Model config: {cfg}")

    logging.info("Creating WanAnimate pipeline.")

    # Create WanAnimate model (matching generate.py)
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
    temp_file = job_file + ".tmp"
    lock_file = job_file + ".lock"

    logging.info(f"Generation service started. Watching {job_file}")

    while True:
        try:
            time.sleep(args.poll_interval)

            if not os.path.exists(job_file):
                continue

            job_to_process = None

            if rank == 0:
                # Find a preprocessed job and mark it as processing
                # Use atomic write pattern for crash safety
                # Use "a" mode to avoid truncating the lock file which can cause race conditions
                with open(lock_file, "a") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    try:
                        lines = []
                        if os.path.exists(job_file):
                            with open(job_file, "r", encoding="utf-8") as f:
                                lines = [line.strip() for line in f if line.strip()]

                        updated_lines = []
                        job_found = False

                        for line in lines:
                            parts = line.strip().split(args.sep)
                            # Job format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded
                            if not job_found and len(parts) >= 6 and parts[1] == "preprocessed":
                                job_found = True
                                parts[1] = "processing"
                                # Set start_time NOW when entering 'processing' status (HPU generation start)
                                parts[3] = str(int(time.time()))
                                job_to_process = parts
                                updated_lines.append(args.sep.join(map(str, parts)) + "\n")
                            else:
                                updated_lines.append(line + "\n")

                        if job_found:
                            # Write to temp file first
                            with open(temp_file, "w", encoding="utf-8") as f:
                                f.writelines(updated_lines)
                                f.flush()
                                os.fsync(f.fileno())
                            # Atomic rename
                            os.replace(temp_file, job_file)
                    finally:
                        fcntl.flock(lf, fcntl.LOCK_UN)

            # Broadcast job to all ranks
            if world_size > 1:
                job_list = [job_to_process] if rank == 0 else [None]
                dist.broadcast_object_list(job_list, src=0)
                job_to_process = job_list[0]

            if job_to_process:
                # Initialize timing variables before try block to ensure they're always defined
                job_id = None
                generate_start_time = time.time()
                try:
                    # Parse job info
                    # Format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded
                    (job_id, status, generate_duration_str, start_time, end_time, *error_msg_parts) = job_to_process

                    # start_time is when job entered 'processing' status (HPU generation start)
                    generate_start_time = float(start_time) if start_time and start_time != "0" else time.time()

                    job_dir = os.path.join(args.video_dir, job_id)
                    video_path = os.path.join(job_dir, "output.mp4")

                    logging.info(f"Processing job {job_id}")

                    # Load preprocessing info
                    preprocess_info = load_preprocess_info(job_dir)

                    preprocess_path = preprocess_info["preprocess_path"]
                    actual_frame_count = preprocess_info["actual_frame_count"]
                    mode = preprocess_info.get("mode", "animate")
                    shift = preprocess_info.get("shift", 5.0)
                    steps = preprocess_info.get("steps", 20)
                    refert_num = preprocess_info.get("refert_num", 1)
                    seed = preprocess_info.get("seed", 0)

                    logging.info(f"Loaded preprocess info: path={preprocess_path}, frames={actual_frame_count}, mode={mode}")

                    # clip_len is the sliding window size for generation (fixed at 77 from animate-14B config)
                    clip_len = 77

                    # Determine if replace mode
                    replace_flag = (mode == "replace")

                    # Run generation
                    logging.info(f"Running generation for job {job_id}...")

                    video = wan_animate.generate(
                        src_root_path=preprocess_path,
                        replace_flag=replace_flag,
                        clip_len=clip_len,
                        refert_num=int(refert_num),
                        shift=float(shift),
                        sample_solver=args.sample_solver,
                        sampling_steps=int(steps),
                        guide_scale=args.sample_guide_scale,
                        seed=int(seed),
                        offload_model=args.offload_model,
                    )

                    # Synchronize HPU before saving
                    if hasattr(torch, 'hpu'):
                        torch.hpu.synchronize()

                    if dist.is_initialized():
                        dist.barrier()

                    if rank == 0:
                        logging.info(f"Saving generated video to {video_path}")
                        save_video(
                            tensor=video[None],
                            save_file=video_path,
                            fps=cfg.sample_fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )

                        # Merge audio if available
                        has_audio = preprocess_info.get("has_audio", False)
                        if has_audio:
                            audio_path = os.path.join(job_dir, "audio.aac")
                            if os.path.exists(audio_path):
                                # Save video without audio first, then merge
                                video_no_audio_path = os.path.join(job_dir, "output_no_audio.mp4")
                                os.rename(video_path, video_no_audio_path)

                                merge_success = merge_audio_to_video(
                                    video_path=video_no_audio_path,
                                    audio_path=audio_path,
                                    output_path=video_path
                                )

                                if merge_success:
                                    logging.info(f"Audio merged successfully into {video_path}")
                                    # Optionally remove the no-audio version
                                    os.remove(video_no_audio_path)
                                else:
                                    # Fallback: rename back to original if merge failed
                                    logging.warning("Audio merge failed, keeping video without audio.")
                                    os.rename(video_no_audio_path, video_path)
                            else:
                                logging.warning(f"Audio file not found at {audio_path}, skipping audio merge.")

                        generate_end_time = time.time()
                        # Job format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded
                        # generate_duration = HPU generation duration only (end_time - start_time)
                        # start_time = when job entered 'processing' status (HPU generation start)
                        # end_time = HPU generation end
                        job_processed = [
                            job_id,
                            "completed",
                            max(0, int(generate_end_time - generate_start_time)),
                            int(generate_start_time),
                            int(generate_end_time),
                            ""  # No error
                        ]
                        update_job(job_processed, args)
                        logging.info(f"Job {job_id} completed successfully.")

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    logging.error(f"Error processing job {job_id}: {error_msg}")

                    # Update job file FIRST, before attempting barrier sync.
                    # This ensures error is recorded even if barrier hangs.
                    if rank == 0:
                        try:
                            generate_end_time = time.time()
                            # Use user-friendly error message for client
                            # If exception has no message, use generic server error
                            user_error_msg = str(e) if str(e) else "Error occurred in server, check error log"
                            encoded_error = encode_error_msg(user_error_msg)
                            job_processed = [
                                job_id if job_id else "unknown",
                                "error",
                                max(0, int(generate_end_time - generate_start_time)),
                                int(generate_start_time),
                                int(generate_end_time),
                                encoded_error
                            ]
                            update_job(job_processed, args)
                            logging.info(f"Job {job_id} marked as error in job file.")
                        except Exception as update_err:
                            logging.error(f"Failed to update job status to error: {update_err}\n{traceback.format_exc()}")

                    # Synchronize all ranks after error to prevent deadlock
                    # Note: This may hang if not all ranks hit the error, but the error is already recorded above.
                    if dist.is_initialized():
                        try:
                            dist.barrier()
                        except Exception as barrier_err:
                            logging.error(f"Barrier failed after error: {barrier_err}")

        except Exception as e:
            logging.error(f"Generation service encountered an error: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    args = _parse_args()
    run_generation_service(args)
