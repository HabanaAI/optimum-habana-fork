import argparse
import os
import threading
import uuid
import pytz
import io
from datetime import datetime
from comps.cores.mega.logger import CustomLogger
from comps.cores.mega.constants import ServiceType
from comps.cores.mega.micro_service import opea_microservices, register_microservice
from comps.cores.mega.base_statistics import statistics_dict, register_statistics
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from fastapi import Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
import time
import torch
import torchaudio
import librosa
import soundfile
from fastapi import File, Form
from pydantic import BaseModel, NonNegativeFloat
from typing import Optional

import random
import numpy as np
from omegaconf import OmegaConf
from api_model import API_Model, auto_prompt_type

class AudioSpeechRequest:
    def __init__(
        self,
        prompt_audio: File = File(None),
        descriptions: str = Form(...),
        gt_lyric: str = Form(...),
        auto_prompt_audio_type: str = Form(...),
        gen_type: Optional[str] = Form(...),
        model: Optional[str] = Form("SongGeneration"),
        seed: Optional[int] = Form(0),
    ):
        self.prompt_audio = prompt_audio
        self.descriptions = descriptions
        self.gt_lyric = gt_lyric
        self.gen_type = gen_type
        self.auto_prompt_audio_type = auto_prompt_audio_type
        self.model = model
        self.seed = seed


class AudioSpeechOutput(BaseModel):
    id: str
    model: str = None
    status: str
    progress: int
    created_time: str
    started_time: str
    finished_time: str
    queue_length: int
    error: str = ""

def _parse_args():
    parser = argparse.ArgumentParser(
        description="CosyVoice API server"
    )
    parser.add_argument('--server-port', type=int, default=8481, help='Demo server port.')
    args = parser.parse_args()
    return args

cmd_args = _parse_args()

logger = CustomLogger("songgeneration")
shanghai_timezone = pytz.timezone('Asia/Shanghai')
lock = threading.Lock()
request_queue = []

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_thread():

    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))

    # 解析命令行参数
    ckpt_path="songgeneration_base_new"
    api_model = API_Model(ckpt_path=ckpt_path, use_flash_attn=False)
    # loop
    print("start pthread")
    while True:
        target_task=None
        with lock:
            for task in request_queue:
                if task['status'] == 'queued':
                    task['status'] = 'processing'
                    task['started_time'] = (datetime.now(shanghai_timezone)).strftime("%Y-%m-%d %H:%M:%S")
                    target_task = task
                    break
        if target_task is None:
            time.sleep(0.1)
            continue
        print("process ", target_task)
        # output_path = './asset/zero_shot_prompt.wav'
        task_id = target_task['task_id']
        output_path = f'tmp/{task_id}/output.wav'
        if task['item']['gen_type'] == 'separate':
            bgm_path = f'tmp/{task_id}/output_bgm.wav'
            vocal_path = f'tmp/{task_id}/output_vocal.wav'
        else:
            bgm_path = None
            vocal_path = None
        set_all_random_seed(task['item']['seed'])
        ret = api_model.generate(item=task['item'],
                                 output_path=output_path,
                                 bgm_path=bgm_path,
                                 vocal_path=vocal_path)

        with lock:
            for task in request_queue:
                if task['task_id'] == target_task['task_id']:
                    if ret is not None:
                        task['status'] = "error"
                        task['error_message'] = ret
                    else:
                        task['status'] = 'completed'
                        task['result_file'] = output_path
                        task['bgm_file'] = bgm_path
                        task['vocal_file'] = vocal_path
                        task['finished_time'] = (datetime.now(shanghai_timezone)).strftime("%Y-%m-%d %H:%M:%S")
                        task["progress"] = 100
                    break

async def resolve_request(request: Request):
    form = await request.form()
    common_args = {
        "prompt_audio": form.get("prompt_audio", ""),
        "descriptions": form.get("descriptions", None),
        "gt_lyric": form.get("gt_lyric", None),
        "gen_type": form.get("gen_type", "mixed"),
        "auto_prompt_audio_type": form.get("auto_prompt_audio_type", None),
        "model": form.get("model", "SongGeneration"),
        "seed": int(form.get("seed", 0)),
    }
    return AudioSpeechRequest(**common_args)


# generate
@register_microservice(
    name="opea_service@songgeneration",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/song",
    host="0.0.0.0",
    port=cmd_args.server_port,
    input_datatype=AudioSpeechRequest,
    output_datatype=AudioSpeechOutput,
)
@register_statistics(names=["opea_service@songgeneration"])
async def songgeneration(input: AudioSpeechRequest = Depends(resolve_request)):
    stream = False
    prompt_audio = None
    task_id = str(uuid.uuid1())[:17]
    tmp_path = f"tmp/{task_id}"
    os.makedirs(tmp_path, exist_ok=True)
    print("prompt_audio ", input.prompt_audio)
    error_message = None
    if input.prompt_audio:
        audio_path = os.path.join(tmp_path, input.prompt_audio.filename)
        contents = await input.prompt_audio.read()
        with open(audio_path, "wb") as af:
            af.write(contents)
        prompt_audio = audio_path
        duration = librosa.get_duration(path=audio_path)
        if duration >= 30:
            error_message = "prompt_audio should be less than 30s"

    if input.gt_lyric is None or input.gt_lyric == '':
        error_message = "Please input gt_lyric"
    if input.auto_prompt_audio_type and not input.auto_prompt_audio_type in auto_prompt_type:
        error_message = "invalid audio prompt type"
    if input.auto_prompt_audio_type and prompt_audio:
        error_message = "auto_prompt_audio_type and prompt_audio cannot be used together"
    if input.gen_type not in ['mixed', 'separate']:
        error_message = f"invalid gen_type {input.gen_type}"

    if error_message is not None:
        content = {
            "error": {
                "message": error_message,
                "code": "400"
            }
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    item = {
        "gt_lyric": input.gt_lyric,
        "seed": input.seed,
        "gen_type": input.gen_type,
    }
    if prompt_audio:
        item["prompt_audio_path"] = prompt_audio
    if input.descriptions:
        item["descriptions"] = input.descriptions
    if input.auto_prompt_audio_type:
        item["auto_prompt_audio_type"] = input.auto_prompt_audio_type

    created_time = (datetime.now(shanghai_timezone)).strftime("%Y-%m-%d %H:%M:%S")
    task = {
        "task_id": task_id,
        "model": input.model,
        "item": item,
        "status": "queued",
        "created_time": created_time,
        "started_time": '',
        "finished_time": '',
        "progress": 0,
        "result_file": None,
        "error_message": None,
        "stream": stream,
        "stream_buf": [],
    }
    with lock:
        request_queue.append(task)
        queue_length = 0
        for task in request_queue:
            if task["status"] in ["queued","processing"]:
                queue_length += 1

    return AudioSpeechOutput(
        id=task_id,
        model=input.model,
        status="queued",
        progress=0,
        created_time=created_time,
        started_time='',
        finished_time='',
        queue_length=queue_length,
    )


#  query information
@register_microservice(
    name="opea_service@songgeneration",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/song/{task_id}",
    host="0.0.0.0",
    port=cmd_args.server_port,
    output_datatype=AudioSpeechOutput,
    methods=['GET']
)
@register_statistics(names=["opea_service@songgeneration"])
async def get_task_information(task_id: str):
    target_task = None
    with lock:
        queue_length = 0
        for task in request_queue:
            if task["status"] in ["queued","processing"]:
                queue_length += 1
            if task['task_id'] == task_id:
                target_task = task.copy()
                break

    if target_task is None:
        content = {
            "error": {
                "message": f"task {task_id} is not found.",
                "code": "400"
            }
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    return AudioSpeechOutput(
        id=task_id,
        model=target_task['model'],
        status=target_task['status'],
        progress=target_task['progress'],
        created_time=target_task['created_time'],
        started_time=target_task['started_time'],
        finished_time=target_task['finished_time'],
        queue_length=queue_length,
        stream=True if target_task['stream'] else False,
    )

#  delete task
@register_microservice(
    name="opea_service@songgeneration",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/song/{task_id}/delete",
    host="0.0.0.0",
    port=cmd_args.server_port,
    output_datatype=AudioSpeechOutput,
    methods=['GET']
)
@register_statistics(names=["opea_service@songgeneration"])
async def delete_task(task_id: str):
    error_message = f"task {task_id} is not found.",
    with lock:
        target_task = None
        for task in request_queue:
            if task['task_id'] == task_id and task['status'] in ['queued']:
                task['status'] = 'deleted'
                target_task = task
                break
            if task['task_id'] == task_id and task['status'] in ['processing']:
                error_message = f"task {task_id} is being processed and cannot be deleted"
                break

    if target_task is None:
        content = {
            "error": {
                "message": error_message,
                "code": "400"
            }
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    content = {
        "success": {
            "message": f"task {task_id} is deleted",
            "code": "200",
        }
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)

# get content
@register_microservice(
    name="opea_service@songgeneration",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/song/{task_id}/content",
    host="0.0.0.0",
    port=cmd_args.server_port,
    methods=['GET']
)
@register_statistics(names=["opea_service@songgeneration"])
async def get_content(task_id: str):
    with lock:
        target_task = None
        for task in request_queue:
            if task['task_id'] == task_id:
                target_task = task.copy()
                break

    error_message = None
    if target_task is None:
        error_message = f"task {task_id} is not found."
    elif target_task['status'] not in ['completed']:
        error_message = f"task {task_id} is not completed."
    if error_message is not None:
        content = {
            "error": {
                "message": error_message,
                "code": "400"
            }
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    return FileResponse(target_task['result_file'], media_type="audio/wav", filename=f"{task_id}.wav")

# get content
@register_microservice(
    name="opea_service@songgeneration",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/song/{task_id}/vocal/content",
    host="0.0.0.0",
    port=cmd_args.server_port,
    methods=['GET']
)
@register_statistics(names=["opea_service@songgeneration"])
async def get_content(task_id: str):
    with lock:
        target_task = None
        for task in request_queue:
            if task['task_id'] == task_id:
                target_task = task.copy()
                break

    error_message = None
    if target_task is None:
        error_message = f"task {task_id} is not found."
    elif target_task['item']['gen_type'] != 'separate':
        error_message = f"task {task_id} is not in separate gen_type"
    elif target_task['status'] not in ['completed']:
        error_message = f"task {task_id} is not completed."
    if error_message is not None:
        content = {
            "error": {
                "message": error_message,
                "code": "400"
            }
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    return FileResponse(target_task['vocal_file'], media_type="audio/wav", filename=f"{task_id}.wav")

# get content
@register_microservice(
    name="opea_service@songgeneration",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/song/{task_id}/bgm/content",
    host="0.0.0.0",
    port=cmd_args.server_port,
    methods=['GET']
)
@register_statistics(names=["opea_service@songgeneration"])
async def get_content(task_id: str):
    with lock:
        target_task = None
        for task in request_queue:
            if task['task_id'] == task_id:
                target_task = task.copy()
                break

    error_message = None
    if target_task is None:
        error_message = f"task {task_id} is not found."
    elif target_task['item']['gen_type'] != 'separate':
        error_message = f"task {task_id} is not in separate gen_type"
    elif target_task['status'] not in ['completed']:
        error_message = f"task {task_id} is not completed."
    if error_message is not None:
        content = {
            "error": {
                "message": error_message,
                "code": "400"
            }
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    return FileResponse(target_task['bgm_file'], media_type="audio/wav", filename=f"{task_id}.wav")

# get content
@register_microservice(
    name="opea_service@songgeneration",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/song/query/auto_prompt_audio_type",
    host="0.0.0.0",
    port=cmd_args.server_port,
    methods=['GET']
)
@register_statistics(names=["opea_service@songgeneration"])
async def get_pretrained_tone():
    content = {
        "success": {
            "message": f"{auto_prompt_type}",
            "code": "200",
        }
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


if __name__ == "__main__":
    logger.info("songgeneration server started.")
    process_p = threading.Thread(target=generate_thread)
    process_p.start()
    opea_microservices["opea_service@songgeneration"].start()
