# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import random
import json
import fcntl
import librosa

from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Union
from fastapi import Form, File, UploadFile
from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry

logger = CustomLogger("opea_Text2Video")


class ServiceType(Enum):
    """The enum of a service type."""
    TEXT2VIDEO = 1


class Text2VideoInput:
    def __init__(
        self,
        prompt: str = Form(None),
        input_reference: Optional[UploadFile] = File(None),
        audio: Union[UploadFile, List[UploadFile]] = File(None),
        audio_guide_scale: Optional[float] = Form(5.0),
        audio_type: Optional[str] = Form("add"),
        model: Optional[str] = Form(None),
        seconds: Optional[int] = Form(4),
        fps: Optional[int] = Form(25),
        shift: Optional[float] = Form(5.0),
        steps: Optional[int] = Form(40),
        seed: Optional[int] = Form(42),
        guide_scale: Optional[float] = Form(5.0),
        size: Optional[str] = Form("720x1280"),
        logo_video: Optional[bool] = Form("False")
    ):
        self.prompt = prompt
        self.input_reference = input_reference
        self.audio = audio
        self.audio_guide_scale = audio_guide_scale
        self.audio_type = audio_type
        self.model = model
        self.seconds = seconds
        self.fps = fps
        self.shift = shift
        self.steps = steps
        self.seed = seed
        self.guide_scale = guide_scale
        self.size = size
        self.logo_video = logo_video


class Text2VideoOutput(BaseModel):
    id: str
    object: str = "video"
    model: str = None
    status: str
    progress: int
    created_at: int
    estimated_time: int
    queue_length: int
    duration: int
    seconds: str
    error: str = ""


def get_audio_duration(file_path):
    return librosa.get_duration(path=file_path)


@OpeaComponentRegistry.register("OPEA_TEXT2VIDEO")
class OpeaText2Video(OpeaComponent):
    """A specialized Text2Video component for video generation."""

    def __init__(
        self,
        name: str,
        description: str,
        config: dict = None,
        video_dir: str = "/home/user/video"
    ):
        """
        Initializes the OpeaText2Video component.

        Args:
            name (str): The name of the component.
            description (str): A description of the component.
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(name, ServiceType.TEXT2VIDEO.name.lower(), description, config)
        self.video_dir = video_dir
        os.makedirs(self.video_dir, exist_ok=True)
        if not self.check_health():
            logger.error("OpeaText2Video health check failed upon initialization.")

    async def invoke(self, input: Text2VideoInput) -> Text2VideoOutput:
        """
        Generates a video based on the provided text prompt.

        Args:
            input (Text2VideoInput): The input data containing the prompt and other parameters.
        """
        created = time.time()
        job_id = f"video_{int(created)}_{random.randint(1000, 9999)}"
        job_dir = os.path.join(self.video_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        input_json = os.path.join(job_dir, "input.json")
        input_json_content = {}

        if input.prompt and len(input.prompt) > 0:
            input_json_content["prompt"] = input.prompt

        if input.audio_type:
            input_json_content["audio_type"] = input.audio_type

        if input.input_reference:
            image_file = os.path.join(job_dir, input.input_reference.filename)
            input_json_content["cond_video"] = image_file
            contents = await input.input_reference.read()
            with open(image_file, "wb") as img_f:
                img_f.write(contents)

        audio_durations = []
        if input.audio and isinstance(input.audio, list):
            audio = {}
            for idx, audio_file in enumerate(input.audio):
                audio_path = os.path.join(job_dir, audio_file.filename)
                audio[f"person{idx+1}"] = audio_path
                contents = await audio_file.read()
                with open(audio_path, "wb") as audio_f:
                    audio_f.write(contents)
                audio_durations.append(get_audio_duration(audio_path))

            input_json_content["cond_audio"] = audio

        with open(input_json, "w") as f:
            json.dump(input_json_content, f, indent=4)

        seconds = int(min(audio_durations)) if audio_durations else 20
        logger.info(f"set audio seconds to {seconds} and audio durations for job {job_id}: {audio_durations}")
        if seconds <= 0:
            raise ValueError("The provided audio files have non-positive durations.")

        status = "queued"
        quality = "standard"
        generate_duration = 0
        start_time = 0
        end_time = 0
        job = [
            job_id,
            status,
            int(created),
            input.prompt,
            seconds,
            input.size,
            quality,
            input.fps,
            input.shift,
            input.steps,
            input.guide_scale,
            input.audio_guide_scale,
            input.seed,
            input.logo_video,
            generate_duration,
            start_time,
            end_time,
            ""
        ]

        sep = os.getenv("SEP", "##$##")
        line = sep.join(map(str, job)) + "\n"
        job_file = os.path.join(self.video_dir, "job.txt")
        with open(job_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        logger.info(f"Job {job_id} queued with prompt: {input.prompt}")
        return job_id

    def check_health(self) -> bool:
        """
        Checks if the model pipeline is initialized.

        Returns:
            bool: True if the pipeline is ready, False otherwise.
        """
        return True
