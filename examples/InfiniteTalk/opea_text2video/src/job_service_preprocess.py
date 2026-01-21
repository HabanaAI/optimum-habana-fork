# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Preprocessing service for Wan2.2 Animate video generation.

This service runs on CPU and handles the preprocessing phase:
- Picks up jobs with status 'queued'
- Marks them as 'preprocessing'
- Runs pose detection, SAM2, and other preprocessing
- Saves preprocessing metadata to preprocess_info.json
- Marks jobs as 'preprocessed' (or 'error' if fails)

Usage:
    python job_service_preprocess.py --video_dir /path/to/videos --process_ckpt_dir /path/to/checkpoints
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
import subprocess
import traceback

from decord import VideoReader

warnings.filterwarnings('ignore')


def extract_audio(video_path: str, output_audio_path: str, duration: float = None) -> bool:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to the input video file
        output_audio_path: Path to save the extracted audio
        duration: Optional duration in seconds to truncate the audio

    Returns:
        bool: True if audio was successfully extracted, False otherwise
    """
    try:
        # Build ffmpeg command to extract audio
        cmd = ["ffmpeg", "-y", "-i", video_path]
        if duration is not None and duration > 0:
            cmd.extend(["-t", str(duration)])
        cmd.extend(["-vn", "-acodec", "aac", "-b:a", "128k", output_audio_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and os.path.exists(output_audio_path):
            # Check if the audio file has content (not empty)
            if os.path.getsize(output_audio_path) > 0:
                logging.info(f"Audio extracted successfully to {output_audio_path}")
                return True
            else:
                os.remove(output_audio_path)
                logging.info("Extracted audio file is empty, driving video has no audio track.")
                return False
        else:
            # No audio track or extraction failed
            logging.info(f"No audio track found or extraction failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logging.warning("Audio extraction timed out.")
        return False
    except Exception as e:
        logging.warning(f"Failed to extract audio: {e}")
        return False


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


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocessing service for Wan2.2 Animate video generation"
    )
    parser.add_argument(
        "--process_ckpt_dir",
        type=str,
        default="/hf/Wan2.2-Animate-14B/process_checkpoint",
        help="Path to preprocessing model checkpoints."
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
    assert args.process_ckpt_dir is not None, "Please specify the process checkpoint directory."
    return args


def _init_logging():
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )


def init_process_pipeline(args):
    """
    Initialize the preprocessing pipeline once at service startup.

    This loads the pose detection, SAM2, and other models into memory
    so they can be reused across multiple jobs.

    Args:
        args: Command line arguments containing process_ckpt_dir

    Returns:
        ProcessPipeline instance
    """
    # Import preprocessing modules from Wan2.2
    wan_root = os.getenv("WAN_ROOT", "/home/user/Wan2.2")
    preprocess_path = os.path.join(wan_root, "wan/modules/animate/preprocess")
    if preprocess_path not in sys.path:
        sys.path.insert(0, preprocess_path)
    from process_pipepline import ProcessPipeline

    # Setup checkpoint paths (matching preprocess_data.py)
    pose2d_checkpoint_path = os.path.join(args.process_ckpt_dir, "pose2d/vitpose_h_wholebody.onnx")
    det_checkpoint_path = os.path.join(args.process_ckpt_dir, "det/yolov10m.onnx")
    # Load SAM2 checkpoint for replace mode support
    sam2_checkpoint_path = os.path.join(args.process_ckpt_dir, "sam2/sam2_hiera_large.pt")
    # FLUX is disabled by default (as in preprocess_data.py default)
    flux_kontext_path = None

    logging.info("Initializing preprocessing pipeline (one-time initialization)...")
    process_pipeline = ProcessPipeline(
        det_checkpoint_path=det_checkpoint_path,
        pose2d_checkpoint_path=pose2d_checkpoint_path,
        sam_checkpoint_path=sam2_checkpoint_path,
        flux_kontext_path=flux_kontext_path
    )
    logging.info("Preprocessing pipeline initialized successfully.")

    return process_pipeline


def run_preprocessing(process_pipeline, job_dir: str, input_data: dict) -> tuple:
    """
    Run preprocessing pipeline for animate job.

    This function runs the preprocessing pipeline from Wan2.2.
    The preprocessing extracts pose, face, and optionally mask/background
    from the driving video and reference image.

    Args:
        process_pipeline: Pre-initialized ProcessPipeline instance
        job_dir: Directory for job files
        input_data: Input parameters from input.json

    Returns:
        tuple: (preprocess_output_path, actual_frame_count)
    """
    video_path = input_data["video_path"]
    image_path = input_data["image_path"]
    mode = input_data.get("mode", "animate")
    size = input_data.get("size", "832*480")
    seconds = input_data.get("seconds", None)  # None means use full video length

    # Parse resolution from size string (e.g., "832*480" -> [832, 480])
    width, height = map(int, size.split("*"))

    replace_flag = (mode == "replace")

    # Determine FPS (fixed at 30 for animate-14B)
    fps = 30

    # If seconds is specified, truncate the driving video before preprocessing
    actual_video_path = video_path
    if seconds is not None and seconds > 0:
        logging.info(f"Truncating driving video to {seconds} seconds...")
        truncated_video_path = os.path.join(job_dir, "truncated_driving.mp4")
        try:
            # Use ffmpeg to truncate video
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-t", str(seconds),
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                truncated_video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and os.path.exists(truncated_video_path):
                actual_video_path = truncated_video_path
                logging.info(f"Video truncated successfully to {truncated_video_path}")
            else:
                logging.warning(f"Failed to truncate video: {result.stderr}. Using original video.")
        except Exception as e:
            logging.warning(f"Failed to truncate video: {e}. Using original video.")

    # Output path for preprocessed data
    preprocess_output = os.path.join(job_dir, "preprocess")
    os.makedirs(preprocess_output, exist_ok=True)

    # Run preprocessing
    # The process_pipepline.py handles:
    # - For animate mode: pose retargeting with retarget_flag=True
    # - For replace mode: mask generation with replace_flag=True
    #
    # Preprocessing parameters (matching the shell scripts):
    # - animate: --retarget_flag
    # - replace: --replace_flag --iterations 3 --k 7 --w_len 1 --h_len 1
    logging.info(f"Running preprocessing: mode={mode}, size={size}, fps={fps}")

    process_pipeline(
        video_path=actual_video_path,
        refer_image_path=image_path,
        output_path=preprocess_output,
        resolution_area=[width, height],
        fps=fps,
        iterations=3,  # Default for replace mode (as in replace_preprocess.sh)
        k=7,           # Default for replace mode
        w_len=1,
        h_len=1,
        retarget_flag=(mode == "animate"),  # Enable retargeting for animate mode
        use_flux=False,
        replace_flag=replace_flag
    )

    # Count actual frames generated (from src_pose.mp4)
    src_pose_path = os.path.join(preprocess_output, "src_pose.mp4")
    if os.path.exists(src_pose_path):
        vr = VideoReader(src_pose_path)
        actual_frame_count = len(vr)
    else:
        # Fallback: estimate from truncated video or original
        actual_frame_count = fps * seconds if seconds else 0

    return preprocess_output, actual_frame_count


def save_preprocess_info(job_dir: str, preprocess_path: str, actual_frame_count: int, input_data: dict, has_audio: bool = False):
    """
    Save preprocessing metadata to preprocess_info.json.

    Args:
        job_dir: Directory for job files
        preprocess_path: Path to preprocessed data
        actual_frame_count: Number of frames in preprocessed video
        input_data: Original input parameters
        has_audio: Whether audio was extracted from driving video
    """
    preprocess_info = {
        "preprocess_path": preprocess_path,
        "actual_frame_count": actual_frame_count,
        "mode": input_data.get("mode", "animate"),
        "size": input_data.get("size", "832*480"),
        "shift": input_data.get("shift", 5.0),
        "steps": input_data.get("steps", 20),
        "refert_num": input_data.get("refert_num", 1),
        "seed": input_data.get("seed", 0),
        "has_audio": has_audio,
    }

    preprocess_info_path = os.path.join(job_dir, "preprocess_info.json")
    with open(preprocess_info_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_info, f, indent=2)

    logging.info(f"Saved preprocessing info to {preprocess_info_path}")


def run_preprocess_service(args):
    """Main preprocessing service loop."""
    _init_logging()

    logging.info(f"Preprocessing service args: {args}")

    # Initialize preprocessing pipeline once at startup
    logging.info("Initializing preprocessing pipeline...")
    process_pipeline = init_process_pipeline(args)

    job_file = os.path.join(args.video_dir, "job_animate.txt")
    temp_file = job_file + ".tmp"
    lock_file = job_file + ".lock"

    logging.info(f"Preprocessing service started. Watching {job_file}")

    while True:
        try:
            time.sleep(args.poll_interval)

            if not os.path.exists(job_file):
                continue

            job_to_process = None

            # Find a queued job and mark it as preprocessing
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
                        if not job_found and len(parts) >= 6 and parts[1] == "queued":
                            job_found = True
                            parts[1] = "preprocessing"
                            # Don't set start_time here - it will be set when entering 'processing' status
                            # start_time is only for HPU generation phase
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

            if job_to_process:
                try:
                    # Parse job info
                    # Format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded
                    (job_id, status, generate_duration_str, start_time, end_time, *error_msg_parts) = job_to_process

                    job_dir = os.path.join(args.video_dir, job_id)
                    os.makedirs(job_dir, exist_ok=True)

                    input_json_path = os.path.join(job_dir, "input.json")

                    with open(input_json_path, "r", encoding="utf-8") as f:
                        input_data = json.load(f)

                    mode = input_data.get("mode", "animate")
                    size = input_data.get("size", "832*480")
                    seconds = input_data.get("seconds")

                    logging.info(f"Processing job {job_id}: mode={mode}, size={size}, seconds={seconds}")

                    # Run preprocessing
                    logging.info(f"Running preprocessing for job {job_id}...")
                    preprocess_path, actual_frame_count = run_preprocessing(process_pipeline, job_dir, input_data)
                    logging.info(f"Preprocessing completed: {preprocess_path}, frames={actual_frame_count}")

                    # Extract audio from driving video
                    video_path = input_data["video_path"]
                    audio_path = os.path.join(job_dir, "audio.aac")
                    audio_duration = input_data.get("seconds")  # None means full duration
                    has_audio = extract_audio(video_path, audio_path, duration=audio_duration)
                    logging.info(f"Audio extraction result: has_audio={has_audio}")

                    # Save preprocessing info for generation service
                    save_preprocess_info(job_dir, preprocess_path, actual_frame_count, input_data, has_audio=has_audio)

                    # Mark job as preprocessed
                    # Don't set start_time here - it will be set when entering 'processing' status
                    # start_time is only for HPU generation phase
                    job_processed = [
                        job_id,
                        "preprocessed",
                        "0",  # generate_duration not yet known
                        "0",  # start_time will be set when entering 'processing' status
                        "0",  # end_time not yet known
                        ""    # No error
                    ]
                    update_job(job_processed, args)
                    logging.info(f"Job {job_id} preprocessing completed, marked as preprocessed.")

                except Exception as e:
                    error_msg = f"{e}\n{traceback.format_exc()}"
                    logging.error(f"Error preprocessing job {job_id}: {error_msg}")

                    error_time = int(time.time())
                    # Encode error message to handle special characters
                    encoded_error = encode_error_msg(str(e))
                    # For errors during preprocessing, start_time and end_time are the same (no HPU time)
                    job_processed = [
                        job_id,
                        "error",
                        "0",  # No HPU generation duration
                        error_time,  # Use error time as placeholder
                        error_time,
                        encoded_error
                    ]
                    update_job(job_processed, args)

        except Exception as e:
            logging.error(f"Preprocessing service encountered an error: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    args = _parse_args()
    run_preprocess_service(args)
