import argparse
import os
import threading
import uuid
import pytz
from datetime import datetime
from comps.cores.mega.logger import CustomLogger
from comps.cores.mega.constants import ServiceType
from comps.cores.mega.micro_service import opea_microservices, register_microservice
from comps.cores.mega.base_statistics import statistics_dict, register_statistics
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from fastapi import Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse
import time
import torch
import torchaudio
import librosa
from fastapi import File, Form
from pydantic import BaseModel, NonNegativeFloat
from typing import Optional

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()


from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav,  logging

class AudioSpeechRequest:
    def __init__(
        self,
        text: str = Form(...),
        mode: str = Form(...),
        pretrained_tone: str = Form(...),
        prompt_text: str = Form(...),
        prompt_audio: str = File(None),
        instruct_text: str = Form(...),
        model: Optional[str] = Form("iic/CosyVoice2-0.5B"),
        speed: Optional[NonNegativeFloat] = Form(1.0),
        seed: Optional[int] = Form(0),
    ):
        self.text = text
        self.mode = mode
        self.pretrained_tone = pretrained_tone
        self.prompt_text = prompt_text
        self.prompt_audio = prompt_audio
        self.instruct_text = instruct_text
        self.model = model
        self.speed = speed
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

logging.getLogger('numba').setLevel(logging.WARNING)
logger = CustomLogger("text2audio")
shanghai_timezone = pytz.timezone('Asia/Shanghai')
lock = threading.Lock()
request_queue = []
max_val = 0.8
prompt_sr = 16000
sft_spk = []

'''
'预训练音色': 'pretrain',
'极速复刻': 'zero_shot',
'跨语种复刻': 'cross_lingual',
'自然语言控制': 'instruct',
'''

ch_en_tone = {
    '中文女':"Chinese Female",
    '中文男':"Chinese Male", 
    '日语男':"Japanese Male", 
    '粤语女':"Cantonese Female", 
    '英文女':"English Female", 
    '英文男':"English Male", 
    '韩语女':"Korean Female",
}

en_ch_tone = {
    "Chinese Female":'中文女',
    "Chinese Male":'中文男', 
    "Japanese Male":'日语男', 
    "Cantonese Female":'粤语女', 
    "English Female":'英文女', 
    "English Male":'英文男', 
    "Korean Female":'韩语女',
}


def postprocess(speech, sample_rate, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(sample_rate * 0.2))], dim=1)
    return speech

def generate_audio(cosyvoice, request_args, output_path):
    # tts_text, mode_checkbox_group, sft_dropdown, prompt_text, instruct_text,
    #                seed, stream, speed
    # audio_output from gr.Audio() works in streaming mode. Seems gr.Audion() in streaming mode
    # requies 2+ chunks of data. So we need to yield a dummy chunk first when stream is False
    tts_text = request_args['text']
    mode_checkbox_group = request_args['mode']
    sft_dropdown = request_args['tone_id']
    prompt_text = request_args['prompt_text']
    instruct_text = request_args['instruct_text']
    seed = request_args['seed']
    speed = request_args['speed']
    prompt_wav = request_args['prompt_audio']

    audio_cat = None
    if mode_checkbox_group == 'pretrain':
        set_all_random_seed(seed)
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=sft_dropdown, stream=False, speed=speed)):
            if i == 0:
                audio_cat = j['tts_speech']
            else:
                audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
    elif mode_checkbox_group == 'zero_shot':
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr), cosyvoice.sample_rate)
        set_all_random_seed(seed)
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=speed)):
            if i == 0:
                audio_cat = j['tts_speech']
            else:
                audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
    elif mode_checkbox_group == 'cross_lingual':
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr), cosyvoice.sample_rate)
        set_all_random_seed(seed)
        for i, j in enumerate(cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False, speed=speed)):
            if i == 0:
                audio_cat = j['tts_speech']
            else:
                audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
    else:
        set_all_random_seed(seed)
        prompt_speech_16k = load_wav(prompt_wav, 16000)
        for i, j in enumerate(cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=False, speed=speed)):
            if i == 0:
                audio_cat = j['tts_speech']
            else:
                audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)

    if audio_cat is not None:
        torchaudio.save(output_path, audio_cat, cosyvoice.sample_rate)
        return None
    else:
        return "error"


def generate_thread():
    
    # init
    try:
        cosyvoice = CosyVoice('pretrained_models/CosyVoice2-0.5B')
    except Exception:
        try:
            cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
        except Exception:
            raise TypeError('no valid model_type!')

    global sft_spk
    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']

    device = 'hpu'
    model = cosyvoice.model.llm.llm.model.bfloat16().eval().to(device)
    cosyvoice.model.llm.llm.model = wrap_in_hpu_graph(model)

    model = cosyvoice.model.llm.llm_decoder.bfloat16().eval().to(device)
    cosyvoice.model.llm.llm_decoder = wrap_in_hpu_graph(model)

    cosyvoice.model.flow = cosyvoice.model.flow.bfloat16().eval()#.to(device)

    #warmup
    prompt_wav = "./asset/9_ZH.wav"
    tts_text = "If one knows how to be grateful and content with small things, then he is a happy person."
    prompt_text = "如果能对小事感到感激和满足，那他就是幸福的人。"
    prompt_speech_16k = postprocess(load_wav(prompt_wav, 16000), cosyvoice.sample_rate)
    set_all_random_seed(0)
    for _ in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
        continue

    # loop
    print("start pthread")
    while True:
        target_task=None
        with lock:
            for task in request_queue:
                if task['status'] == 'queued':
                    task['status'] = 'processing'
                    task['started_time'] = (datetime.now(shanghai_timezone)).strftime("%Y-%m-%d %H:%M:%S")
                    target_task = task.copy()
                    break
        if target_task is None:
            time.sleep(1)
            continue
        print("process ", target_task)
        # output_path = './asset/zero_shot_prompt.wav'
        task_id = task['task_id']
        output_path = f'tmp/{task_id}/output.wav'
        ret = generate_audio(cosyvoice, task['request_args'], output_path)
        with lock:
            for task in request_queue:
                if task['task_id'] == target_task['task_id']:
                    if ret is not None:
                        task['status'] = "error"
                        task['error_message'] = ret
                    else:
                        task['status'] = 'completed'
                        task['result_file'] = output_path
                        task['finished_time'] = (datetime.now(shanghai_timezone)).strftime("%Y-%m-%d %H:%M:%S")
                        task["progress"] = 100
                    break
        time.sleep(5)


async def resolve_request(request: Request):
    form = await request.form()
    common_args = {
        "text": form.get("text", ""),
        "mode": form.get("mode", None),
        "pretrained_tone": form.get("pretrained_tone", None),
        "prompt_text": form.get("prompt_text", None),
        "prompt_audio": form.get("prompt_audio", None),
        "instruct_text": form.get("instruct_text", None),
        "model": form.get("model", "iic/CosyVoice2-0.5B"),
        "speed": float(form.get("speed", 1.0)),
        "seed": int(form.get("seed", 0)),
    }
    return AudioSpeechRequest(**common_args)


# generate
@register_microservice(
    name="opea_service@text2audio",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/speech",
    host="0.0.0.0",
    port=cmd_args.server_port,
    input_datatype=AudioSpeechRequest,
    output_datatype=AudioSpeechOutput,
)
@register_statistics(names=["opea_service@text2audio"])
async def text2audio(input: AudioSpeechRequest = Depends(resolve_request)):
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

    if input.text is None or input.text == '':
        error_message = "Please input text"
    if input.mode is None or input.mode == '':
        error_message = "Please input mode"
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if input.mode in ['instruct']:
        if input.instruct_text is None or input.instruct_text == '':
            error_message = "You are using instruct mode, please input instruct_text"
        if prompt_audio is None:
            error_message = "You are using instruct mode, please input prompt_audio"
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if input.mode in ['zero_shot', 'cross_lingual']:
        if prompt_audio is None:
            error_message = f"You are using {input.mode} mode please input prompt_audio"
        elif torchaudio.info(prompt_audio).sample_rate < prompt_sr:
            error_message = f'prompt sample rate {torchaudio.info(prompt_audio).sample_rate} lower than {prompt_sr}'
    # sft mode only use sft_dropdown
    if input.mode in ['pretrain']:
        if input.pretrained_tone is None or input.pretrained_tone == '':
            error_message = "You are using pretrain mode please input pretrained_tone"
        if input.pretrained_tone not in en_ch_tone and input.pretrained_tone not in ch_en_tone:
            error_message = "invalid pretrained_tone"
    # zero_shot mode only use prompt_wav prompt text
    if input.mode in ['zero_shot']:
        if input.prompt_text is None or input.prompt_text == '':
            error_message = "Please input prompt_text"

    if error_message is not None:
        content = {
            "error": {
                "message": error_message,
                "code": "400"
            }
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    if input.pretrained_tone in en_ch_tone:
        input.pretrained_tone = en_ch_tone[input.pretrained_tone]
    
    request_args = {
        "text": input.text,
        "mode": input.mode,
        "tone_id": input.pretrained_tone,
        "prompt_text": input.prompt_text,
        "prompt_audio": prompt_audio,
        "instruct_text": input.instruct_text,
        "speed": input.speed,
        "seed": input.seed,
    }
    created_time = (datetime.now(shanghai_timezone)).strftime("%Y-%m-%d %H:%M:%S")
    task = {
        "task_id": task_id,
        "model": input.model,
        "request_args": request_args,
        "status": "queued",
        "created_time": created_time,
        "started_time": '',
        "finished_time": '',
        "progress": 0,
        "result_file": None,
        "error_message": None,
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
    name="opea_service@text2audio",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/speech/{task_id}",
    host="0.0.0.0",
    port=cmd_args.server_port,
    output_datatype=AudioSpeechOutput,
    methods=['GET']
)
@register_statistics(names=["opea_service@text2audio"])
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
    )

#  delete task
@register_microservice(
    name="opea_service@text2audio",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/speech/{task_id}/delete",
    host="0.0.0.0",
    port=cmd_args.server_port,
    output_datatype=AudioSpeechOutput,
    methods=['GET']
)
@register_statistics(names=["opea_service@text2audio"])
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
    name="opea_service@text2audio",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/speech/{task_id}/content",
    host="0.0.0.0",
    port=cmd_args.server_port,
    methods=['GET']
)
@register_statistics(names=["opea_service@text2audio"])
async def get_content(task_id: str):
    with lock:
        target_task = None
        for task in request_queue:
            if task['task_id'] == task_id:
                target_task = task.copy()

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
    name="opea_service@text2audio",
    service_type=ServiceType.TTS,
    endpoint="/v1/audio/speech/query/pretrained_tone",
    host="0.0.0.0",
    port=cmd_args.server_port,
    methods=['GET']
)
@register_statistics(names=["opea_service@text2audio"])
async def get_pretrained_tone():
    en_tone = [ch_en_tone[t] for t in sft_spk]
    content = {
        "success": {
            "message": f"{en_tone}",
            "code": "200",
        }
    } 
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


if __name__ == "__main__":
    logger.info("Text2Audio server started.")
    process_p = threading.Thread(target=generate_thread)
    process_p.start()
    opea_microservices["opea_service@text2audio"].start()