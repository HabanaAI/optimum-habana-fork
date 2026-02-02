# Install
```shell
cd Qwen3-TTS
pip install -r requirement-gaudi.txt
pip install -e .
```


# Prepare model

```shell
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz  --local_dir ./Qwen3-TTS-Tokenizer-12Hz 
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./Qwen3-TTS-12Hz-0.6B-Base
```


# Simple Commad Demo
```shell
wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-0115/APS-en_29.wav -O clone.wav
PT_HPU_LAZY_MODE=1 python custom_voice_demo.py
PT_HPU_LAZY_MODE=1 python voice_clone_demo.py
PT_HPU_LAZY_MODE=1 python voice_design_clone_demo.py
PT_HPU_LAZY_MODE=1 python voice_design_demo.py

```
