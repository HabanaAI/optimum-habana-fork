# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
import fcntl
import shutil
import math
import json

from fastapi import Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from component_animate import (
    AnimateInput,
    AnimateOutput,
    ServiceType,
    OpeaAnimate,
    SUPPORTED_ANIMATE_SIZES,
    decode_error_msg,
)


# Initialize logger and component loader
logger = CustomLogger("animate")
component_loader = None
LOGFLAG = os.getenv("LOGFLAG", "False").lower() in ("true", "1", "t")


def validate_form_parameters(form, files):
    """Validate and convert form parameters to their expected types."""
    try:
        # Check required files
        if "image" not in files or files["image"] is None:
            raise ValueError("Missing required parameter: image")
        if "video" not in files or files["video"] is None:
            raise ValueError("Missing required parameter: video")

        # Get optional parameters with defaults
        mode = form.get("mode", "animate")
        if mode not in ["animate", "replace"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'animate' or 'replace'.")

        size = form.get("size", "832*480")
        if size not in SUPPORTED_ANIMATE_SIZES:
            raise ValueError(f"Invalid size: {size}. Supported: {SUPPORTED_ANIMATE_SIZES}")

        refert_num = int(form.get("refert_num", 1))
        if refert_num not in [1, 5]:
            raise ValueError(f"Invalid refert_num: {refert_num}. Must be 1 or 5.")

        # seconds is optional - None means use full driving video length
        seconds_str = form.get("seconds", None)
        seconds = int(seconds_str) if seconds_str is not None and seconds_str != "" else None
        if seconds is not None and seconds <= 0:
            raise ValueError("seconds must be greater than 0 or omitted (for full video length).")

        params = {
            "image": files["image"],
            "video": files["video"],
            "mode": mode,
            "size": size,
            "seconds": seconds,
            "refert_num": refert_num,
            "seed": int(form.get("seed", -1)),
            "shift": float(form.get("shift", 5.0)),
            "steps": int(form.get("steps", 20)),
        }

        return params, None
    except (ValueError, TypeError) as e:
        error_content = {"error": {"message": f"{e}", "code": "400"}}
        return None, JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_content)


async def resolve_request(request: Request):
    form = await request.form()

    # Extract files from form
    files = {
        "image": form.get("image"),
        "video": form.get("video"),
    }

    validated_params, error_response = validate_form_parameters(form, files)
    if error_response:
        return error_response
    return AnimateInput(**validated_params)


def estimate_queue_time(seconds: int, steps: int, mode: str = "animate", size: str = "832*480", include_preprocess_buffer: bool = False) -> int:
    """
    Estimate generation time in minutes for Animate-14B.
    Based on actual benchmark results from Gaudi accelerators with 8 HPU cards.
    Scales with actual rank_size from RANK_SIZE environment variable.

    Args:
        seconds: Video duration in seconds (0 or empty means unknown/full video)
        steps: Diffusion sampling steps
        mode: "animate" or "replace"
        size: Video resolution (e.g., "832*480", "1280*720")
        include_preprocess_buffer: Whether to include preprocessing time buffer
            (True for waiting statuses: queued, preprocessing, preprocessed)

    Returns:
        Estimated time in minutes
    """
    # Get the actual number of HPU cards being used
    rank_size = int(os.getenv("RANK_SIZE", 1))

    # If seconds is 0 or unknown, assume average video length of 3 seconds
    # Note: With ffprobe integration, we now detect actual video duration,
    # so this fallback is rarely used
    effective_seconds = seconds if seconds and seconds > 0 else 3

    # Resolution multiplier based on benchmark data
    # Standard resolutions (832*480, 480*832): 1.0x
    # HD resolutions (1280*720, 720*1280): ~2.5x
    resolution_multiplier = 1.0
    if size in ["1280*720", "720*1280"]:
        resolution_multiplier = 2.5

    # Benchmark data shows linear relationship between steps and time
    # For animate mode at 832*480 with 8 HPUs:
    # - 3.54s: steps=20→131s (37.0s per second)
    # - 8.58s: steps=20→230s (26.8s per second)
    # - 10.24s: steps=20→281s (27.4s per second)

    if mode == "animate":
        # Base time per second of video (varies by duration due to clip overhead)
        if effective_seconds <= 2:
            # ~40s per second at steps=20 (extrapolated)
            base_time_per_sec = 2.0 * steps
        elif effective_seconds <= 4:
            # ~37s per second at steps=20 (from 3.54s benchmark)
            base_time_per_sec = 1.85 * steps
        elif effective_seconds <= 6:
            # ~32s per second at steps=20 (interpolated)
            base_time_per_sec = 1.6 * steps
        else:
            # ~27s per second at steps=20 (from 8.58s, 10.24s benchmarks)
            base_time_per_sec = 1.35 * steps

        total_time_sec = base_time_per_sec * effective_seconds
    else:
        # Replace mode: ~1.67-1.75x slower than animate mode
        # Scaled from new animate benchmarks

        if effective_seconds <= 2:
            # ~67s per second at steps=20
            base_time_per_sec = 3.35 * steps
        elif effective_seconds <= 4:
            # ~62s per second at steps=20
            base_time_per_sec = 3.1 * steps
        elif effective_seconds <= 6:
            # ~54s per second at steps=20
            base_time_per_sec = 2.7 * steps
        else:
            # ~45s per second at steps=20
            base_time_per_sec = 2.25 * steps

        total_time_sec = base_time_per_sec * effective_seconds

    # Apply resolution multiplier
    total_time_sec *= resolution_multiplier

    # Scale by rank_size (benchmark data is for 8 cards, so scale inversely)
    # More cards = faster processing, fewer cards = slower processing
    total_time_sec = total_time_sec * 8 / rank_size

    # Add preprocessing buffer if in waiting status
    # Preprocessing runs on CPU before HPU generation
    if include_preprocess_buffer:
        if mode == "animate":
            total_time_sec += 10  # ~10s preprocessing for animate mode
        else:
            total_time_sec += 30  # ~30s preprocessing for replace mode

    # Convert to minutes and round up
    return max(1, math.ceil(total_time_sec / 60))


def load_input_json(job_id: str) -> dict:
    """
    Load input.json for a job to get parameters.

    Args:
        job_id: The job ID

    Returns:
        Dictionary with job parameters, or empty dict if not found
    """
    input_json_path = os.path.join(os.getenv("VIDEO_DIR"), job_id, "input.json")
    if os.path.exists(input_json_path):
        try:
            with open(input_json_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def calculate_progress(job_info: list, input_data: dict) -> tuple:
    """
    Calculate job progress and remaining time.

    Only calculates progress for 'processing' status (HPU generation phase).
    For waiting statuses (queued, preprocessing, preprocessed), returns 0% progress.

    Args:
        job_info: Job information list [job_id, status, generate_duration, start_time, end_time, error_msg_encoded]
        input_data: Parameters loaded from input.json

    Returns:
        Tuple of (progress percentage, remaining time in minutes)
    """
    status = job_info[1]
    seconds = input_data.get("seconds") or input_data.get("effective_seconds", 0) or 0
    steps = input_data.get("steps", 20)
    mode = input_data.get("mode", "animate")
    size = input_data.get("size", "832*480")

    # For waiting statuses, return 0% progress with full estimated time (including preprocess buffer)
    if status in ("queued", "preprocessing", "preprocessed"):
        estimated_time = estimate_queue_time(seconds, steps, mode, size, include_preprocess_buffer=True)
        return 0, estimated_time

    # For processing status, calculate progress based on HPU generation time only
    # start_time is set when job enters 'processing' status
    estimated_time = estimate_queue_time(seconds, steps, mode, size, include_preprocess_buffer=False)
    start_time = int(job_info[3]) if job_info[3] else 0
    elapsed_time = int(time.time()) - start_time
    progress = int(min(int((elapsed_time / (estimated_time * 60)) * 100), 99))
    left_time = int(max(1, int(estimated_time - (elapsed_time / 60))))
    return progress, left_time


def generate_response(video_id: str) -> AnimateOutput:
    """
    Generate response for a video job.

    Args:
        video_id: The job ID to look up

    Returns:
        AnimateOutput with job status
    """
    job_file = os.path.join(os.getenv("VIDEO_DIR"), "job_animate.txt")
    lock_file = job_file + ".lock"

    if os.path.exists(job_file):
        sep = os.getenv("SEP", ",")
        queue_estimated_time_in_minutes = 0
        queue_length = 0
        job_info = None
        job_input_data = None

        # Use lock file for consistency with write operations
        # Use "a" mode to avoid truncating the lock file which can cause race conditions
        with open(lock_file, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)  # Shared lock for read-only
            try:
                with open(job_file, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    job = line.strip().split(sep)

                    # New format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded
                    if len(job) < 6:
                        continue

                    current_job_id = job[0]
                    current_input_data = load_input_json(current_job_id)

                    if current_job_id == video_id:
                        job_info = job
                        job_input_data = current_input_data
                        break

                    # Count jobs ahead in queue (all active statuses except completed/error)
                    if job[1] in ("queued", "preprocessing", "preprocessed", "processing"):
                        queue_length += 1
                        seconds = current_input_data.get("seconds") or current_input_data.get("effective_seconds", 0) or 0
                        steps = current_input_data.get("steps", 20)
                        mode = current_input_data.get("mode", "animate")
                        size = current_input_data.get("size", "832*480")
                        if job[1] == "processing":
                            _, left_time = calculate_progress(job, current_input_data)
                            queue_estimated_time_in_minutes += left_time
                        else:
                            # Include preprocess buffer for waiting statuses
                            queue_estimated_time_in_minutes += estimate_queue_time(seconds, steps, mode, size, include_preprocess_buffer=True)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        if job_info:
            # Job format: job_id,status,generate_duration,start_time,end_time,error_msg_encoded
            # Parameters (seconds, steps, etc.) come from input.json
            created_at = job_input_data.get("created_at", 0)
            # Use user-specified seconds first, fall back to effective_seconds (auto-detected)
            display_seconds = job_input_data.get("seconds") or job_input_data.get("effective_seconds")
            seconds_str = str(int(display_seconds)) if display_seconds is not None else ""

            if job_info[1] == "processing":
                progress, left_time = calculate_progress(job_info, job_input_data)
                return AnimateOutput(
                    id=job_info[0],
                    model=os.getenv("MODEL", "Wan2.2-Animate-14B"),
                    status=job_info[1],
                    progress=progress,
                    created_at=created_at,
                    seconds=seconds_str,
                    duration=0,
                    estimated_time=left_time,
                    queue_length=0,
                    error=""
                )
            elif job_info[1] in ("queued", "preprocessing", "preprocessed"):
                # For waiting statuses, calculate estimated time for current job (with preprocess buffer)
                # and add to queue time for jobs ahead
                _, current_job_estimated_time = calculate_progress(job_info, job_input_data)
                total_estimated_time = queue_estimated_time_in_minutes + current_job_estimated_time
                return AnimateOutput(
                    id=job_info[0],
                    model=os.getenv("MODEL", "Wan2.2-Animate-14B"),
                    status=job_info[1],
                    progress=0,
                    created_at=created_at,
                    seconds=seconds_str,
                    duration=0,
                    estimated_time=int(total_estimated_time),
                    queue_length=queue_length,
                    error=""
                )
            else:
                # Decode error message if status is error
                error_msg = ""
                if job_info[1] == "error" and len(job_info) > 5:
                    error_msg = decode_error_msg(job_info[5])

                return AnimateOutput(
                    id=job_info[0],
                    model=os.getenv("MODEL", "Wan2.2-Animate-14B"),
                    status=job_info[1],
                    progress=100 if job_info[1] == "completed" else 0,
                    created_at=created_at,
                    seconds=seconds_str,
                    duration=int(job_info[2]) if job_info[1] == "completed" else 0,
                    estimated_time=0 if job_info[1] == "completed" else int(queue_estimated_time_in_minutes),
                    queue_length=0 if job_info[1] == "completed" else queue_length,
                    error=error_msg
                )

    content = {
        "error": {
            "message": f"Video with id {video_id} not found.",
            "code": "404"
        }
    }
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)


@register_microservice(
    name="opea_service@animate",
    service_type=ServiceType.ANIMATE,
    endpoint="/v1/animate",
    host="0.0.0.0",
    port=9397,
    input_datatype=AnimateInput,
    output_datatype=AnimateOutput,
)
@register_statistics(names=["opea_service@animate"])
async def animate(input_data: AnimateInput = Depends(resolve_request)) -> AnimateOutput:
    """
    Process an animate video generation request.

    Args:
        input_data (AnimateInput): The input data containing image, video, and parameters.

    Returns:
        AnimateOutput: The result of the video generation.
    """
    if isinstance(input_data, JSONResponse):
        return input_data

    start = time.time()
    if component_loader:
        try:
            job_id = await component_loader.invoke(input_data)
            results = generate_response(job_id)
        except ValueError as ve:
            error_content = {"error": {"message": str(ve), "code": "400"}}
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_content)
        except Exception as e:
            error_content = {"error": {"message": f"Internal server error: {e}", "code": "500"}}
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_content)
    else:
        raise RuntimeError("Component loader is not initialized.")

    latency = time.time() - start
    statistics_dict["opea_service@animate"].append_latency(latency, None)
    return results


@register_microservice(
    name="opea_service@animate",
    service_type=ServiceType.ANIMATE,
    endpoint="/v1/animate/{video_id}",
    host="0.0.0.0",
    port=9397,
    methods=["GET"],
)
@register_statistics(names=["opea_service@animate"])
async def get_animate_status(video_id: str):
    """Get the status of an animate job."""
    try:
        return generate_response(video_id)
    except Exception as e:
        error_content = {"error": {"message": f"Internal server error: {e}", "code": "500"}}
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_content)


@register_microservice(
    name="opea_service@animate",
    service_type=ServiceType.ANIMATE,
    endpoint="/v1/animate/{video_id}",
    host="0.0.0.0",
    port=9397,
    methods=["DELETE"],
)
@register_statistics(names=["opea_service@animate"])
async def delete_animate(video_id: str):
    """Cancel/delete an animate job."""
    try:
        job_file = os.path.join(os.getenv("VIDEO_DIR"), "job_animate.txt")
        temp_file = job_file + ".tmp"
        lock_file = job_file + ".lock"

        if not os.path.exists(job_file):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": {"message": f"Job queue is missing and video with id {video_id} not found.", "code": "404"}},
            )

        sep = os.getenv("SEP", ",")
        deleted_job_info = None
        updated_lines = []
        job_found = False

        # Use atomic write pattern for crash safety
        # Use "a" mode to avoid truncating the lock file which can cause race conditions
        with open(lock_file, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(job_file, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    job = line.strip().split(sep)
                    if job[0] == video_id:
                        job_found = True
                        if job[1] == "processing":
                            return JSONResponse(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                content={"error": {"message": f"Video with id {video_id} is processing and cannot be deleted.", "code": "400"}},
                            )
                        deleted_job_info = job
                    else:
                        updated_lines.append(line)

                if not job_found:
                    return JSONResponse(
                        status_code=status.HTTP_404_NOT_FOUND,
                        content={"error": {"message": f"Video with id {video_id} not found.", "code": "404"}},
                    )

                # Write to temp file first, then atomic rename
                with open(temp_file, "w") as f:
                    f.writelines(updated_lines)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_file, job_file)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        if deleted_job_info:
            # Load input.json to get parameters for response
            deleted_input_data = load_input_json(deleted_job_info[0])
            created_at = deleted_input_data.get("created_at", 0)
            # Use user-specified seconds first, fall back to effective_seconds (auto-detected)
            display_seconds = deleted_input_data.get("seconds") or deleted_input_data.get("effective_seconds")
            seconds_str = str(int(display_seconds)) if display_seconds is not None else ""

            video_folder_path = os.path.join(os.getenv("VIDEO_DIR"), deleted_job_info[0])
            if os.path.isdir(video_folder_path):
                shutil.rmtree(video_folder_path)
            return AnimateOutput(
                id=deleted_job_info[0],
                model=os.getenv("MODEL", "Wan2.2-Animate-14B"),
                status="deleted",
                progress=0,
                created_at=created_at,
                seconds=seconds_str,
                duration=int(deleted_job_info[2]) if deleted_job_info[2] else 0,
                estimated_time=0,
                queue_length=0,
                error=""
            )

    except Exception as e:
        error_content = {"error": {"message": f"Internal server error: {e}", "code": "500"}}
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_content)


@register_microservice(
    name="opea_service@animate",
    service_type=ServiceType.ANIMATE,
    endpoint="/v1/animate/{video_id}/content",
    host="0.0.0.0",
    port=9397,
    methods=["GET"],
)
@register_statistics(names=["opea_service@animate"])
async def get_animate_content(video_id: str):
    """Download the generated video file."""
    try:
        res = generate_response(video_id)
        if isinstance(res, JSONResponse):
            return res
        if res.status == "completed":
            video_path = os.path.join(os.getenv("VIDEO_DIR"), video_id, "output.mp4")
            if os.path.exists(video_path):
                return FileResponse(video_path, media_type="video/mp4", filename=f"{video_id}.mp4")
            else:
                error_content = {"error": {"message": f"Video file for id {video_id} not found.", "code": "404"}}
                return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=error_content)
        else:
            return res
    except Exception as e:
        error_content = {"error": {"message": f"Internal server error: {e}", "code": "500"}}
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_content)


def main():
    """
    Main function to set up and run the animate microservice.
    """
    global component_loader

    parser = argparse.ArgumentParser(description="Wan Animate Microservice")
    parser.add_argument("--model_name_or_path", type=str, default="Wan2.2-Animate-14B", help="Model name or path.")
    parser.add_argument("--rank_size", type=int, default=1, help="Determines how many ranks are divided into context parallel group.")
    parser.add_argument("--video_dir", type=str, default="/home/user/video", help="Video output directory.")
    parser.add_argument("--sep", type=str, default=",", help="Separator for job attributes.")

    args = parser.parse_args()
    os.environ["MODEL"] = args.model_name_or_path
    os.environ["RANK_SIZE"] = str(args.rank_size)
    os.environ["VIDEO_DIR"] = args.video_dir
    os.environ["SEP"] = args.sep

    animate_component_name = os.getenv("ANIMATE_COMPONENT_NAME", "OPEA_ANIMATE")

    try:
        component_loader = OpeaComponentLoader(
            component_name=animate_component_name,
            description=f"OPEA ANIMATE Component: {animate_component_name}",
            config=args.__dict__,
            video_dir=args.video_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize component loader: {e}")
        exit(1)

    logger.info("Animate service started.")
    opea_microservices["opea_service@animate"].start()


if __name__ == "__main__":
    main()
