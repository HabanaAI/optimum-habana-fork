import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import time

model = Qwen3TTSModel.from_pretrained(
    "./Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="hpu",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
torch.set_printoptions(precision=10)
# single inference
tt = time.time()
wavs, sr = model.generate_custom_voice(
    text="She said she would be here by noon.",
    language="English", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Ryan",
    instruct="Very happy.", # Omit if not needed.
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("output_custom_voice2.wav", wavs[0], sr)

tt = time.time()
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("output_custom_voice.wav", wavs[0], sr)


# batch inference
tt = time.time()
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."],
)
print("-------------------------total inference time:", time.time() - tt)
tt = time.time()
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."],
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)