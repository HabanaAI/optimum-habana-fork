import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "./Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

ref_audio = "./clone.wav"
ref_text  = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

import time
tt = time.time()
wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
print("-------------------------total inference time:", time.time() - tt)
tt = time.time()
wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("output_voice_clone.wav", wavs[0], sr)
