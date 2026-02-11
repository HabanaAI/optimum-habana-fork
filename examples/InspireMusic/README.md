# Install

```shell
cd InspireMusic
apt update
apt install sox libsox-dev ffmpeg
pip install -r requirements-gaudi.txt
```


# Prepare model
```shell
mkdir -p pretrained_models
# Download models
# ModelScope
git clone https://www.modelscope.cn/iic/InspireMusic-1.5B-Long.git pretrained_models/InspireMusic-1.5B-Long
# HuggingFace
git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git pretrained_models/InspireMusic-1.5B-Long
```


# Simple Commad Demo
```shell
cd example/music_generation
PT_HPU_LAZY_MODE=1 python text-to-music.py
```

# batch inference
```shell
cd example/music_generation
PT_HPU_LAZY_MODE=1 bash batch_infer_1.5b_long.sh
```