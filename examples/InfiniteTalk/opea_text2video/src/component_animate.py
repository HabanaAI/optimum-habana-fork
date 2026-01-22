# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import random
import json
import fcntl
import base64
import subprocess
import math

from enum import Enum
from pydantic import BaseModel
from typing import Optional
from fastapi import Form, File, UploadFile
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry

logger = CustomLogger("opea_Animate")


class ServiceType(Enum):
    """The enum of a service type."""
    ANIMATE = 1


class AnimateInput:
    """Input parameters for Wan Animate service."""

    def __init__(
        self,
        image: UploadFile = File(...),
        video: UploadFile = File(...),
        mode: Optional[str] = Form("animate"),
        size: Optional[str] = Form("832*480"),
        seconds: Optional[int] = Form(None),
        refert_num: Optional[int] = Form(1),
        seed: Optional[int] = Form(-1),
        shift: Optional[float] = Form(5.0),
        steps: Optional[int] = Form(20),
    ):
        self.image = image
        self.video = video
        self.mode = mode
        self.size = size
        self.seconds = seconds  # None means use full driving video length
        self.refert_num = refert_num
        self.seed = seed
        self.shift = shift
        self.steps = steps


class AnimateOutput(BaseModel):
    """Output response for Wan Animate service."""
    id: str
    object: str = "video"
    model: str = "Wan2.2-Animate-14B"
    status: str
    progress: int
    created_at: int
    estimated_time: int
    queue_length: int
    duration: int
    seconds: str
    error: str = ""


# Supported sizes for Animate-14B (720P and 480P)
SUPPORTED_ANIMATE_SIZES = ["1280*720", "720*1280", "832*480", "480*832"]


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


def decode_error_msg(encoded_msg: str) -> str:
    """
    Decode base64 encoded error message.

    Args:
        encoded_msg: Base64 encoded error message

    Returns:
        Decoded error message string, or empty string if input is empty
    """
    if not encoded_msg:
        return ""
    try:
        return base64.b64decode(encoded_msg.encode('ascii')).decode('utf-8')
    except Exception:
        return encoded_msg  # Return as-is if decoding fails


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, Exception) as e:
        logger.warning(f"Failed to get video duration: {e}")
    return 0.0


def calculate_frame_num(seconds: int, fps: int = 30) -> int:
    """
    Calculate frame number from seconds.
    Frame number must be 4n+1 for Animate-14B.

    Args:
        seconds: Video duration in seconds
        fps: Frames per second (default 30)

    Returns:
        Frame number adjusted to 4n+1
    """
    frame_num = fps * seconds
    # Adjust to nearest valid value (4n+1)
    frame_num = ((frame_num - 1) // 4) * 4 + 1
    return frame_num


@OpeaComponentRegistry.register("OPEA_ANIMATE")
class OpeaAnimate(OpeaComponent):
    """A specialized Animate component for video generation from reference image and driving video."""

    def __init__(
        self,
        name: str,
        description: str,
        config: dict = None,
        video_dir: str = "/home/user/video"
    ):
        """
        Initializes the OpeaAnimate component.

        Args:
            name (str): The name of the component.
            description (str): A description of the component.
            config (dict, optional): Configuration dictionary. Defaults to None.
            video_dir (str): Output directory for generated videos.
        """
        super().__init__(name, ServiceType.ANIMATE.name.lower(), description, config)
        self.video_dir = video_dir
        os.makedirs(self.video_dir, exist_ok=True)
        if not self.check_health():
            logger.error("OpeaAnimate health check failed upon initialization.")

    async def invoke(self, input: AnimateInput) -> str:
        """
        Creates an animate job based on the provided inputs.

        Args:
            input (AnimateInput): The input data containing image, video, and parameters.

        Returns:
            str: Job ID for tracking the generation
        """
        created = time.time()
        job_id = f"video_{int(created)}_{random.randint(1000, 9999)}"
        job_dir = os.path.join(self.video_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Validate parameters
        if input.mode not in ["animate", "replace"]:
            raise ValueError(f"Invalid mode: {input.mode}. Must be 'animate' or 'replace'.")

        if input.size not in SUPPORTED_ANIMATE_SIZES:
            raise ValueError(f"Invalid size: {input.size}. Supported: {SUPPORTED_ANIMATE_SIZES}")

        if input.refert_num not in [1, 5]:
            raise ValueError(f"Invalid refert_num: {input.refert_num}. Must be 1 or 5.")

        if input.seconds is not None and input.seconds <= 0:
            raise ValueError("seconds must be greater than 0 or None (for full video length).")

        if input.shift < 1.0:
            raise ValueError(f"Invalid shift: {input.shift}. Must be >= 1.0 (recommended: 3.0-8.0).")

        # Save input files
        image_path = os.path.join(job_dir, input.image.filename)
        image_contents = await input.image.read()
        with open(image_path, "wb") as f:
            f.write(image_contents)

        video_path = os.path.join(job_dir, input.video.filename)
        video_contents = await input.video.read()
        with open(video_path, "wb") as f:
            f.write(video_contents)

        # Get actual video duration using ffprobe
        video_duration = get_video_duration(video_path)
        video_duration_rounded = round(video_duration, 2) if video_duration > 0 else None

        # Determine effective seconds: use user-specified or actual video duration
        effective_seconds = input.seconds
        if effective_seconds is None and video_duration_rounded is not None:
            effective_seconds = video_duration_rounded

        # Create input.json (stores all job parameters)
        input_json_content = {
            "image_path": image_path,
            "video_path": video_path,
            "mode": input.mode,
            "size": input.size,
            "seconds": input.seconds,  # User-specified seconds (can be None)
            "video_duration": video_duration_rounded,  # Actual video duration in seconds (rounded to 2 decimal places)
            "effective_seconds": effective_seconds,  # Used for estimation (user-specified or actual)
            "refert_num": input.refert_num,
            "seed": input.seed if input.seed >= 0 else random.randint(0, 2**32 - 1),
            "shift": input.shift,
            "steps": input.steps,
            "created_at": int(created),
        }

        input_json_path = os.path.join(job_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_json_content, f, indent=4)

        # Create job entry (simplified format)
        # Format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded
        # All other parameters are stored in input.json
        status = "queued"
        generate_duration = 0
        start_time = 0
        end_time = 0
        error_msg = ""

        job = [
            job_id,
            status,
            generate_duration,
            start_time,
            end_time,
            error_msg  # Will be base64 encoded when there's an actual error
        ]

        sep = os.getenv("SEP", ",")
        line = sep.join(map(str, job)) + "\n"
        job_file = os.path.join(self.video_dir, "job_animate.txt")
        lock_file = job_file + ".lock"

        # Use the same lock file as update operations for consistency
        # Use "a" mode to avoid truncating the lock file which can cause race conditions
        with open(lock_file, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(job_file, "a") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        logger.info(f"Animate job {job_id} queued with mode: {input.mode}, size: {input.size}")
        return job_id

    def check_health(self) -> bool:
        """
        Checks if the component is healthy.

        Returns:
            bool: True if healthy, False otherwise.
        """
        return True
