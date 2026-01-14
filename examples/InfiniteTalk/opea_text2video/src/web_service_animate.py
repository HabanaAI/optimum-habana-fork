# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
import fcntl
import shutil
import math

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
from component_animate import AnimateInput, AnimateOutput, ServiceType, OpeaAnimate, SUPPORTED_ANIMATE_SIZES


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

        size = form.get("size", "1280*720")
        if size not in SUPPORTED_ANIMATE_SIZES:
            raise ValueError(f"Invalid size: {size}. Supported: {SUPPORTED_ANIMATE_SIZES}")

        refert_num = int(form.get("refert_num", 1))
        if refert_num not in [1, 5]:
            raise ValueError(f"Invalid refert_num: {refert_num}. Must be 1 or 5.")

        seconds = int(form.get("seconds", 2))
        if seconds <= 0:
            raise ValueError("seconds must be greater than 0.")

        params = {
            "image": files["image"],
            "video": files["video"],
            "prompt": form.get("prompt", "视频中的人在做动作"),
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


def estimate_queue_time(seconds: int, steps: int) -> int:
    """
    Estimate generation time in minutes for Animate-14B.

    Args:
        seconds: Video duration in seconds
        steps: Diffusion sampling steps

    Returns:
        Estimated time in minutes
    """
    rank_size = int(os.getenv("RANK_SIZE", 1))
    # Animate-14B is slower than TI2V due to preprocessing + larger model
    # Rough estimate: preprocessing ~30s + generation ~2min per second of video at 20 steps
    preprocessing_time = 0.5  # minutes
    generation_time = seconds * 2.0 * steps * rank_size / (20 * 8)
    return max(1, math.ceil(preprocessing_time + generation_time))


def calculate_progress(job_info: list) -> tuple:
    """
    Calculate job progress and remaining time.

    Args:
        job_info: Job information list

    Returns:
        Tuple of (progress percentage, remaining time in minutes)
    """
    estimated_time = estimate_queue_time(int(job_info[3]), int(job_info[8]))
    start_time = int(job_info[12])
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
    if os.path.exists(job_file):
        sep = os.getenv("SEP", ",")
        queue_estimated_time_in_minutes = 0
        queue_length = 0
        job_info = None

        with open(job_file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                lines = f.readlines()
                for line in lines:
                    job = line.strip().split(sep)

                    if len(job) < 15:
                        continue

                    if job[0] == video_id:
                        job_info = job
                        queue_estimated_time_in_minutes += estimate_queue_time(int(job[3]), int(job[8]))
                        break

                    if job[1] == "queued":
                        queue_length += 1
                        queue_estimated_time_in_minutes += estimate_queue_time(int(job[3]), int(job[8]))

                    if job[1] == "processing":
                        progress, left_time = calculate_progress(job)
                        queue_length += 1
                        queue_estimated_time_in_minutes += left_time
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        if job_info:
            # Job format: id,status,created_time,seconds,size,mode,fps,shift,steps,refert_num,seed,generate_duration,start_time,end_time,error_msg
            if job_info[1] == "processing":
                progress, left_time = calculate_progress(job_info)
                return AnimateOutput(
                    id=job_info[0],
                    model=os.getenv("MODEL", "Wan2.2-Animate-14B"),
                    status=job_info[1],
                    progress=progress,
                    created_at=int(job_info[2]),
                    seconds=job_info[3],
                    duration=0,
                    estimated_time=left_time,
                    queue_length=0,
                    error=""
                )
            else:
                return AnimateOutput(
                    id=job_info[0],
                    model=os.getenv("MODEL", "Wan2.2-Animate-14B"),
                    status=job_info[1],
                    progress=100 if job_info[1] == "completed" else 0,
                    created_at=int(job_info[2]),
                    seconds=job_info[3],
                    duration=int(job_info[11]) if job_info[1] == "completed" else 0,
                    estimated_time=0 if job_info[1] == "completed" else int(queue_estimated_time_in_minutes),
                    queue_length=0 if job_info[1] == "completed" else queue_length,
                    error=job_info[14] if job_info[1] == "error" else ""
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
        if not os.path.exists(job_file):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": {"message": f"Job queue is missing and video with id {video_id} not found.", "code": "404"}},
            )

        sep = os.getenv("SEP", ",")
        deleted_job_info = None
        updated_lines = []
        job_found = False

        with open(job_file, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
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

                # Rewrite the file without the deleted line
                f.seek(0)
                f.truncate()
                f.writelines(updated_lines)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        if deleted_job_info:
            video_folder_path = os.path.join(os.getenv("VIDEO_DIR"), deleted_job_info[0])
            if os.path.isdir(video_folder_path):
                shutil.rmtree(video_folder_path)
            return AnimateOutput(
                id=deleted_job_info[0],
                model=os.getenv("MODEL", "Wan2.2-Animate-14B"),
                status="deleted",
                progress=0,
                created_at=int(deleted_job_info[2]),
                seconds=deleted_job_info[3],
                duration=int(deleted_job_info[11]),
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
