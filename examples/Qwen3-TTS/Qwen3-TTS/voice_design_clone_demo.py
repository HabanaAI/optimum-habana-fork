import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# create a reference audio in the target style using the VoiceDesign model
design_model = Qwen3TTSModel.from_pretrained(
    "./Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

ref_text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
ref_instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
import time
tt = time.time()
ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct,
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("voice_design_reference.wav", ref_wavs[0], sr)

# build a reusable clone prompt from the voice design reference
clone_model = Qwen3TTSModel.from_pretrained(
    "./Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),   # or "voice_design_reference.wav"
    ref_text=ref_text,
)

sentences = [
    "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
    "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
]

# reuse it for multiple single calls
tt = time.time()
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("clone_single_1.wav", wavs[0], sr)

tt = time.time()
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
print("-------------------------total inference time:", time.time() - tt)
sf.write("clone_single_2.wav", wavs[0], sr)

# or batch generate in one call
tt = time.time()
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English", "English"],
    voice_clone_prompt=voice_clone_prompt,
)
print("-------------------------total inference time:", time.time() - tt)
for i, w in enumerate(wavs):
    sf.write(f"clone_batch_{i}.wav", w, sr)
