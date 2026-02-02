import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "./Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

# single inference
import time
tt = time.time()
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
)
print("-------------------------total inference time:", time.time() - tt)
tt = time.time()
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("output_voice_design.wav", wavs[0], sr)

tt = time.time()
wavs, sr = model.generate_voice_design(
    text="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
    language="English",
    instruct="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("output_voice_design2.wav", wavs[0], sr)

# batch inference
tt = time.time()
wavs, sr = model.generate_voice_design(
    text=[
      "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
      "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ],
    language=["Chinese", "English"],
    instruct=[
      "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
      "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ],
)
tt = time.time()
print("-------------------------total inference time:", time.time() - tt)
wavs, sr = model.generate_voice_design(
    text=[
      "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
      "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ],
    language=["Chinese", "English"],
    instruct=[
      "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
      "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ],
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("output_voice_design_1.wav", wavs[0], sr)
sf.write("output_voice_design_2.wav", wavs[1], sr)
