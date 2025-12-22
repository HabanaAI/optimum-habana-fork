
# HPU环境变量设置
import os
os.environ['PT_HPU_LAZY_MODE'] = '1'

import aiohttp
from aiohttp import web
import asyncio
import sys
import torch
import torchaudio
import habana_frameworks.torch as ht_torch
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 全局模型缓存，避免重复加载
MODEL_PATH = "/workspace/data/CosyVoice2-0.5B"
cosyvoice = None

def load_cosyvoice_model():
    global cosyvoice
    if cosyvoice is None:
        print("Loading CosyVoice2 model on HPU ...")
        cosyvoice = CosyVoice2(MODEL_PATH, load_jit=False, load_trt=False, fp16=False)
        device = 'hpu'
        # HPU graph wrap
        model = cosyvoice.model.llm.llm.model.bfloat16().eval().to(device)
        cosyvoice.model.llm.llm.model = wrap_in_hpu_graph(model)
        model = cosyvoice.model.llm.llm_decoder.bfloat16().eval().to(device)
        cosyvoice.model.llm.llm_decoder = wrap_in_hpu_graph(model)
        cosyvoice.model.flow = cosyvoice.model.flow.bfloat16().eval()
        print("Model loaded and moved to HPU.")

def run_cosyvoice_inference(
    input_text="和你聊天真的很开心",
    prompt_text="希望你以后能够做的比我还好呦。",
    voicepath="/CosyVoice/asset/zero_shot_prompt.wav",
    voice="中文女",
    response_format="wav",
    sample_rate=24000,
    stream=False,
    speed=1
):
    load_cosyvoice_model()
    # 处理prompt_audio参数
    prompt_audio = voicepath if voicepath else None
    # 加载并预处理音频
    prompt_speech_16k = None
    if prompt_audio:
        prompt_speech_16k = load_wav(prompt_audio, 16000)
    # 推理
    with torch.no_grad():
        for out in cosyvoice.inference_zero_shot(
            input_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed
        ):
            audio = out['tts_speech']
            break
    # 保存为 wav 文件
    wav_path = "cosyvoice_gen.wav"
    torchaudio.save(wav_path, audio, sample_rate)
    return wav_path

async def handle_speech(request):
    try:
        data = await request.json()
        input_text = data.get("input", "")
        prompt_text = data.get("prompt_text", "")
        voicepath = data.get("voicepath", "")
        voice = data.get("voice", "中文女")
        response_format = data.get("response_format", "wav")
        sample_rate = data.get("sample_rate", 24000)
        stream = data.get("stream", False)
        speed = data.get("speed", 1)

        # 调用 CosyVoice 模型生成音频
        wav_path = await asyncio.to_thread(
            run_cosyvoice_inference,
            input_text=input_text,
            prompt_text=prompt_text,
            voicepath=voicepath,
            voice=voice,
            response_format=response_format,
            sample_rate=sample_rate,
            stream=stream,
            speed=speed
        )

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        return web.Response(body=audio_bytes, content_type="audio/wav")
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

app = web.Application()
app.router.add_post("/v1/audio/speech", handle_speech)

if __name__ == "__main__":
    web.run_app(app, port=8022)