# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import random
import json
import fcntl

from enum import Enum
from pydantic import BaseModel
from typing import Optional, List
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
        prompt: Optional[str] = Form("视频中的人在做动作"),
        mode: Optional[str] = Form("animate"),
        size: Optional[str] = Form("1280*720"),
        seconds: Optional[int] = Form(2),
        refert_num: Optional[int] = Form(1),
        seed: Optional[int] = Form(-1),
        shift: Optional[float] = Form(5.0),
        steps: Optional[int] = Form(20),
    ):
        self.image = image
        self.video = video
        self.prompt = prompt
        self.mode = mode
        self.size = size
        self.seconds = seconds
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

        if input.seconds <= 0:
            raise ValueError("seconds must be greater than 0.")

        # Save input files
        image_path = os.path.join(job_dir, input.image.filename)
        image_contents = await input.image.read()
        with open(image_path, "wb") as f:
            f.write(image_contents)

        video_path = os.path.join(job_dir, input.video.filename)
        video_contents = await input.video.read()
        with open(video_path, "wb") as f:
            f.write(video_contents)

        # Create input.json
        input_json_content = {
            "prompt": input.prompt,
            "image_path": image_path,
            "video_path": video_path,
            "mode": input.mode,
            "size": input.size,
            "seconds": input.seconds,
            "refert_num": input.refert_num,
            "seed": input.seed if input.seed >= 0 else random.randint(0, 2**32 - 1),
            "shift": input.shift,
            "steps": input.steps,
        }

        input_json_path = os.path.join(job_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_json_content, f, indent=4)

        # Create job entry
        # Format: id,status,created_time,seconds,size,mode,fps,shift,steps,refert_num,seed,generate_duration,start_time,end_time,error_msg
        fps = 30  # Fixed for Animate-14B
        status = "queued"
        generate_duration = 0
        start_time = 0
        end_time = 0
        error_msg = ""

        job = [
            job_id,
            status,
            int(created),
            input.seconds,
            input.size,
            input.mode,
            fps,
            input.shift,
            input.steps,
            input.refert_num,
            input_json_content["seed"],  # Use the resolved seed
            generate_duration,
            start_time,
            end_time,
            error_msg
        ]

        sep = os.getenv("SEP", ",")
        line = sep.join(map(str, job)) + "\n"
        job_file = os.path.join(self.video_dir, "job_animate.txt")

        with open(job_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        logger.info(f"Animate job {job_id} queued with mode: {input.mode}, size: {input.size}")
        return job_id

    def check_health(self) -> bool:
        """
        Checks if the component is healthy.

        Returns:
            bool: True if healthy, False otherwise.
        """
        return True
