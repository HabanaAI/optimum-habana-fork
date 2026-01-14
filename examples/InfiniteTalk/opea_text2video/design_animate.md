# Wan Animate OPEA Service Design Document

## Overview

This document describes the design for implementing Wan2.2 Animate as an OPEA microservice. The service enables users to animate a reference image using motion from a driving video.

## Architecture

The service follows a two-part architecture similar to the existing Wan2.2 TI2V service:

```
┌─────────────────────┐     ┌─────────────────────┐
│  web_service_       │     │  job_service_       │
│  animate.py         │────▶│  animate.py         │
│  (FastAPI Server)   │     │  (Background Worker)│
└─────────────────────┘     └─────────────────────┘
         │                           │
         │                           ▼
         │                  ┌─────────────────────┐
         │                  │  1. Preprocess      │
         ▼                  │  2. Generate        │
┌─────────────────────┐     └─────────────────────┘
│  job.txt / JSON     │
│  (Job Queue)        │
└─────────────────────┘
```

### Components

1. **Web Service (`web_service_animate.py`)**:

   - FastAPI-based HTTP server
   - Accepts user requests via REST API
   - Validates input parameters
   - Stores job information in job file and JSON

2. **Job Service (`job_service_animate.py`)**:
   - Background worker process
   - Reads jobs from job queue
   - Executes preprocessing pipeline
   - Runs video generation
   - Updates job status

## API Design

### Endpoint: `POST /v1/animate`

Creates a new animate video generation job.

#### Required Input Parameters

| Parameter | Type | Description                                       |
| --------- | ---- | ------------------------------------------------- |
| `image`   | file | Reference image - the character/person to animate |
| `video`   | file | Driving video - the motion source                 |

#### Optional Input Parameters

| Parameter    | Type   | Default                | Choices                                              | Description                                                                                          |
| ------------ | ------ | ---------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `prompt`     | string | `"视频中的人在做动作"` | -                                                    | Text description for the animation                                                                   |
| `mode`       | string | `"animate"`            | `"animate"`, `"replace"`                             | Animation mode. "animate" transfers motion to reference image; "replace" replaces character in video |
| `size`       | string | `"1280*720"`           | `"1280*720"`, `"720*1280"`, `"832*480"`, `"480*832"` | Output video resolution (720P or 480P)                                                               |
| `seconds`    | int    | `2`                    | -                                                    | Video length in seconds. Internally converted to frame_num (fps=30, must be 4n+1)                    |
| `refert_num` | int    | `1`                    | `1`, `5`                                             | Temporal guidance frames. 1=faster, 5=better temporal consistency                                    |
| `seed`       | int    | `-1`                   | -                                                    | Random seed for reproducibility. -1 means random                                                     |
| `shift`      | float  | `5.0`                  | -                                                    | Noise schedule shift parameter                                                                       |
| `steps`      | int    | `20`                   | -                                                    | Diffusion sampling steps. Higher=better quality but slower                                           |

#### Response

Response body follows the same format as Wan2.2-TI2V-5B service:

| Parameter        | Type   | Description                                                                   |
| ---------------- | ------ | ----------------------------------------------------------------------------- |
| `id`             | string | Unique identifier for the video generation job.                               |
| `object`         | string | Object type, always `"video"`.                                                |
| `model`          | string | Model used for generation (e.g., `"Wan2.2-Animate-14B"`).                     |
| `status`         | string | Current job status (`queued`, `processing`, `completed`, `deleted`, `error`). |
| `progress`       | int    | Approximate completion percentage of the task.                                |
| `created_at`     | int    | Unix timestamp (seconds) when the job was created.                            |
| `estimated_time` | int    | Estimated completion time (minutes).                                          |
| `queue_length`   | int    | Number of jobs queued before this one.                                        |
| `duration`       | int    | Time taken to generate the video (seconds).                                   |
| `seconds`        | int    | Final duration of the generated video (seconds).                              |
| `error`          | string | Message explaining the failure reason (if any).                               |

**Success (202 Accepted)**:

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "queued",
  "progress": 0,
  "created_at": 1767068943,
  "estimated_time": 5,
  "queue_length": 1,
  "duration": 0,
  "seconds": "2",
  "error": ""
}
```

**Error (400 Bad Request)**:

```json
{
  "error": {
    "message": "Missing required parameter: image",
    "code": "400"
  }
}
```

### Endpoint: `GET /v1/animate/{video_id}`

Get the status of an animate job.

#### Response

**Processing**:

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "processing",
  "progress": 45,
  "created_at": 1767068943,
  "estimated_time": 3,
  "queue_length": 0,
  "duration": 0,
  "seconds": "2",
  "error": ""
}
```

**Completed**:

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "completed",
  "progress": 100,
  "created_at": 1767068943,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 120,
  "seconds": "2",
  "error": ""
}
```

**Error**:

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "error",
  "progress": 0,
  "created_at": 1767068943,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 0,
  "seconds": "2",
  "error": "Preprocessing failed: No face detected in reference image"
}
```

### Endpoint: `GET /v1/animate/{video_id}/content`

Download the generated video file (mp4).

### Endpoint: `DELETE /v1/animate/{video_id}`

Cancel/delete an animate job.

#### Response

**Success**:

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "deleted",
  "progress": 0,
  "created_at": 1767068943,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 0,
  "seconds": "2",
  "error": ""
}
```

## Service Startup Configuration

These parameters are set at service startup and are **not user-configurable**:

| Parameter            | Description                                        | Default                                     |
| -------------------- | -------------------------------------------------- | ------------------------------------------- |
| `--ckpt_dir`         | Path to Wan2.2-Animate-14B checkpoint directory    | `/hf/Wan2.2-Animate-14B`                    |
| `--process_ckpt_dir` | Path to preprocessing model checkpoints            | `/hf/Wan2.2-Animate-14B/process_checkpoint` |
| `--ulysses_size`     | Sequence parallelism size for multi-card inference | `1`                                         |
| `--video_dir`        | Output directory for generated videos              | `/home/user/video`                          |
| `--sample_solver`    | Sampling solver algorithm (`unipc` or `dpm++`)     | `dpm++`                                     |
| `--sep`              | Separator for job file fields                      | `,`                                         |

### Preprocessing Options (Fixed Defaults)

| Option          | Value                     | Description                              |
| --------------- | ------------------------- | ---------------------------------------- |
| `retarget_flag` | `True` (for animate mode) | Enable pose retargeting                  |
| `use_flux`      | `False`                   | FLUX image editing (disabled by default) |
| `fps`           | `30`                      | Video frame rate                         |
| `iterations`    | `3`                       | Mask dilation iterations (replace mode)  |
| `k`             | `7`                       | Mask kernel size (replace mode)          |

## Internal Processing Flow

### Job File Format

Each job is stored as a line in `job_animate.txt` with comma-separated values:

```
id,status,created_time,seconds,size,mode,fps,shift,steps,refert_num,seed,generate_duration,start_time,end_time,error_msg
```

### Input JSON Format

For each job, an `input.json` file is created in `{video_dir}/{job_id}/`:

```json
{
  "prompt": "视频中的人在做动作",
  "image_path": "/path/to/reference_image.jpg",
  "video_path": "/path/to/driving_video.mp4",
  "mode": "animate",
  "size": "1280*720",
  "seconds": 2,
  "refert_num": 1,
  "seed": 42,
  "shift": 5.0,
  "steps": 20
}
```

### Job Service Workflow

1. **Poll Job Queue**: Check `job.txt` for queued jobs
2. **Mark Processing**: Update job status to "processing"
3. **Load Input Data**: Read `input.json` for job parameters
4. **Preprocess**:
   - Extract poses from driving video
   - Extract face regions
   - Prepare reference image
   - (For replace mode) Generate masks and backgrounds
5. **Generate**:
   - Load preprocessed data
   - Run diffusion model
   - Save output video
6. **Update Status**: Mark job as "completed" or "error"

## Frame Number Calculation

Video length is specified in seconds and converted to frame_num:

```python
fps = 30  # Fixed for animate-14B
frame_num = fps * seconds
# Adjust to nearest valid value (4n+1)
frame_num = ((frame_num - 1) // 4) * 4 + 1
```

Examples:

- 2 seconds → 60 frames → adjusted to 61 (4×15+1)
- 3 seconds → 90 frames → adjusted to 89 (4×22+1)

## Supported Resolutions

Animate-14B supports both 720P and 480P resolutions:

| Size String  | Width | Height | Resolution | Aspect Ratio     |
| ------------ | ----- | ------ | ---------- | ---------------- |
| `"1280*720"` | 1280  | 720    | 720P       | 16:9 (Landscape) |
| `"720*1280"` | 720   | 1280   | 720P       | 9:16 (Portrait)  |
| `"832*480"`  | 832   | 480    | 480P       | 16:9 (Landscape) |
| `"480*832"`  | 480   | 832    | 480P       | 9:16 (Portrait)  |

## Error Handling

| Error Type                 | HTTP Code | Description                                       |
| -------------------------- | --------- | ------------------------------------------------- |
| Missing required parameter | 400       | prompt, image, or video not provided              |
| Invalid parameter value    | 400       | e.g., invalid size, refert_num not 1 or 5         |
| Job not found              | 404       | video_id does not exist                           |
| Processing error           | 500       | Internal error during preprocessing or generation |

## Dependencies

- Wan2.2 (with HPU support)
- FastAPI
- PyTorch with Habana support
- OpenCV, MoviePy (for video processing)
- ONNX Runtime (for pose detection)

## File Structure

```
opea_text2video/
├── src/
│   ├── web_service_animate.py    # New: FastAPI web service
│   ├── job_service_animate.py    # New: Background job worker
│   ├── component_animate.py      # New: OPEA component definition
│   ├── web_service.py            # Existing: InfiniteTalk/Wan TI2V
│   ├── job_service_wan.py        # Existing: Wan TI2V job worker
│   ├── component.py              # Existing: OPEA components
│   └── util.py                   # Shared utilities
├── Dockerfile-animate            # New: Docker build for animate
├── docker-compose-animate.yml    # New: Docker compose config
├── start_animate.sh              # New: Startup script
└── design_animate.md             # This document
```

## Future Considerations

1. **Caching Preprocessed Data**: Allow reusing preprocessed data for multiple generations with different parameters
2. **Progress Reporting**: More granular progress updates (preprocessing %, generation step %)
3. **Batch Processing**: Support multiple reference images or driving videos in one request
4. **Quality Presets**: Predefined combinations of steps/shift for "fast", "balanced", "quality" modes

## Review Checklist

- [ ] API endpoint design
- [ ] Input parameter definitions
- [ ] Default values
- [ ] Error handling
- [ ] File format specifications
- [ ] Processing workflow
- [ ] Service startup configuration
