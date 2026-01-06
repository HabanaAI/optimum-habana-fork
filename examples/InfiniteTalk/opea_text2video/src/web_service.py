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
from component import Text2VideoInput, Text2VideoOutput, ServiceType, OpeaText2Video


# Initialize logger and component loader
logger = CustomLogger("text2video")
component_loader = None
LOGFLAG = os.getenv("LOGFLAG", "False").lower() in ("true", "1", "t")


def validate_form_parameters(form):
    """Validate and convert form parameters to their expected types."""
    try:
        audio = []
        if "audio[]" in form:
            audio += form.getlist("audio[]")
        elif "audio" in form:
            audio += form.getlist("audio")

        params = {
            "prompt": form.get("prompt"),
            "input_reference": form.get("input_reference"),
            "audio": audio,
            "audio_guide_scale": float(form.get("audio_guide_scale", 5.0)),
            "audio_type": form.get("audio_type", "add"),
            "model": form.get("model"),
            "seconds": int(form.get("seconds", 4)),
            "fps": int(form.get("fps", 25)),
            "shift": float(form.get("shift", 5.0)),
            "steps": int(form.get("steps", 40)),
            "seed": int(form.get("seed", 42)),
            "guide_scale": float(form.get("guide_scale", 5.0)),
            "size": form.get("size", "720x1280"),
            "logo_video": form.get("logo_video", "False")
        }

        if params["seconds"] <= 0:
            raise ValueError("The 'seconds' parameter must be greater than 0.")

        # Validate size format
        width, height = params["size"].split("x")
        if not (width.isdigit() and height.isdigit()):
            raise ValueError("Invalid size format. Expected 'widthxheight'.")

        if not params["input_reference"] or len(params["audio"]) == 0:
            raise ValueError("'input_reference' and 'audio' must be provided.")

        return params, None
    except (ValueError, TypeError) as e:
        error_content = {"error": {"message": f"Invalid parameter type: {e}", "code": "400"}}
        return None, JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_content)


async def resolve_request(request: Request):
    form = await request.form()
    validated_params, error_response = validate_form_parameters(form)
    if error_response:
        return error_response
    return Text2VideoInput(**validated_params)


def calculate_progress(job_info):
    estimated_time = estimate_queue_time(int(job_info[4]), int(job_info[9]))
    start_time = int(job_info[15])
    elapsed_time = int(time.time()) - start_time
    progress = int(min(int((elapsed_time / (estimated_time * 60)) * 100), 99))
    left_time = int(max(1, int(estimated_time - (elapsed_time / 60))))
    return progress, left_time


def estimate_queue_time(seconds, steps):
    steps = max(steps, 1)
    return math.ceil(seconds * 1.16 * steps / 20) if seconds <= 10 else math.ceil(int(seconds * steps / 20)) if seconds <= 15 else math.ceil(int(seconds * 0.83 * steps / 20))


def generate_response(video_id) -> Text2VideoOutput:
    job_file = os.path.join(os.getenv("VIDEO_DIR"), "job.txt")
    if os.path.exists(job_file):
        sep = os.getenv("SEP")
        queue_estimated_time_in_minutes = 0
        queue_length = 0
        job_info = None
        with open(job_file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                lines = f.readlines()
                for line in lines:
                    job = line.strip().split(sep)

                    if len(job) < 17:
                        continue

                    if job[0] == video_id:
                        job_info = job
                        queue_estimated_time_in_minutes += estimate_queue_time(int(job[4]), int(job[9]))
                        break

                    if job[1] == "queued":
                        queue_length += 1
                        queue_estimated_time_in_minutes += estimate_queue_time(int(job[4]), int(job[9]))

                    if job[1] == "processing":
                        progress, left_time = calculate_progress(job)
                        queue_length += 1
                        queue_estimated_time_in_minutes += left_time
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        if job_info:
            if job_info[1] == "processing":
                progress, left_time = calculate_progress(job_info)
                return Text2VideoOutput(
                    id=job_info[0],
                    model=os.getenv("MODEL"),
                    status=job_info[1],
                    progress=progress,
                    created_at=int(job_info[2]),
                    seconds=job_info[4],
                    duration=0,
                    estimated_time=left_time,
                    queue_length=0,
                    error=job_info[-1] if job_info[1] == "error" else ""
                )
            else:
                return Text2VideoOutput(
                    id=job_info[0],
                    model=os.getenv("MODEL"),
                    status=job_info[1],
                    progress=100 if job_info[1] == "completed" else 0,
                    created_at=int(job_info[2]),
                    seconds=job_info[4],
                    duration=job_info[14],
                    estimated_time=0 if job_info[1] == "completed" else int(queue_estimated_time_in_minutes),
                    queue_length=0 if job_info[1] == "completed" else queue_length,
                    error=job_info[-1] if job_info[1] == "error" else ""
                )

    content = {
        "error": {
            "message": f"Video with id {video_id} not found.",
            "code": "404"
        }
    }
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/videos",
    host="0.0.0.0",
    port=9396,
    input_datatype=Text2VideoInput,
    output_datatype=Text2VideoOutput,
)
@register_statistics(names=["opea_service@text2video"])
async def text2video(input_data: Text2VideoInput = Depends(resolve_request)) -> Text2VideoOutput:
    """
    Process a text-to-video generation request.

    Args:
        input_data (Text2VideoInput): The input data containing the prompt.

    Returns:
        Text2VideoOutput: The result of the video generation.
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
    statistics_dict["opea_service@text2video"].append_latency(latency, None)
    return results


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/videos/{video_id}",
    host="0.0.0.0",
    port=9396,
    methods=["GET"],
)
@register_statistics(names=["opea_service@text2video"])
async def get_video(video_id: str):
    try:
        return generate_response(video_id)
    except Exception as e:
        error_content = {"error": {"message": f"Internal server error: {e}", "code": "500"}}
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_content)


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/videos/{video_id}",
    host="0.0.0.0",
    port=9396,
    methods=["DELETE"],
)
@register_statistics(names=["opea_service@text2video"])
async def delete_video(video_id: str):
    try:
        job_file = os.path.join(os.getenv("VIDEO_DIR"), "job.txt")
        if not os.path.exists(job_file):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": {"message": f"Job queue is missing and video with id {video_id} not found.", "code": "404"}},
            )

        sep = os.getenv("SEP")
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
            return Text2VideoOutput(
                id=deleted_job_info[0],
                model=os.getenv("MODEL"),
                status="deleted",
                progress=0,
                created_at=int(deleted_job_info[2]),
                seconds=deleted_job_info[4],
                duration=int(deleted_job_info[14]),
                estimated_time=0,
                queue_length=0,
                error=""
            )

    except Exception as e:
        error_content = {"error": {"message": f"Internal server error: {e}", "code": "500"}}
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_content)


@register_microservice(
    name="opea_service@text2video",
    service_type=ServiceType.TEXT2VIDEO,
    endpoint="/v1/videos/{video_id}/content",
    host="0.0.0.0",
    port=9396,
    methods=["GET"],
)
@register_statistics(names=["opea_service@text2video"])
async def get_video_content(video_id: str):
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
    Main function to set up and run the text-to-video microservice.
    """
    global component_loader

    parser = argparse.ArgumentParser(description="Text-to-Video Microservice")
    parser.add_argument("--model_name_or_path", type=str, default="InfinteTalk", help="Model name or path.")
    parser.add_argument("--video_dir", type=str, default="/home/user/video", help="Video output directory.")

    args = parser.parse_args()
    os.environ["MODEL"] = args.model_name_or_path
    os.environ["VIDEO_DIR"] = args.video_dir
    os.environ["SEP"] = "$###$"
    text2video_component_name = os.getenv("TEXT2VIDEO_COMPONENT_NAME", "OPEA_TEXT2VIDEO")

    try:
        component_loader = OpeaComponentLoader(
            component_name=text2video_component_name,
            description=f"OPEA IMAGES_GENERATIONS Component: {text2video_component_name}",
            config=args.__dict__,
            video_dir=args.video_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize component loader: {e}")
        exit(1)

    logger.info("Text-to-video server started.")
    opea_microservices["opea_service@text2video"].start()


if __name__ == "__main__":
    main()
