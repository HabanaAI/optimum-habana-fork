<p> <a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank"> <img alt="logo" src="./envs/logo.png" width="100%"></a></p>

<p>
 <a href="https://funaudiollm.github.io/inspiremusic" target="_blank"><img alt="Demo" src="https://img.shields.io/badge/Demo-InspireMusic?labelColor=%20%23FDB062&label=InspireMusic&color=%20%23f79009"></a>
<a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank"><img alt="Code" src="https://img.shields.io/badge/Code-InspireMusic?labelColor=%20%237372EB&label=InspireMusic&color=%20%235462eb"></a>
<a href="https://modelscope.cn/models/iic/InspireMusic" target="_blank"><img alt="Model" src="https://img.shields.io/badge/InspireMusic-Model-green"></a>
<a href="https://modelscope.cn/studios/iic/InspireMusic/summary" target="_blank"><img alt="Space" src="https://img.shields.io/badge/Spaces-ModelScope-pink?labelColor=%20%237b8afb&label=Spaces&color=%20%230a5af8"></a>
<a href="https://huggingface.co/spaces/FunAudioLLM/InspireMusic" target="_blank"><img alt="Space" src="https://img.shields.io/badge/HuggingFace-Spaces?labelColor=%20%239b8afb&label=Spaces&color=%20%237a5af8"></a>
<a href="http://arxiv.org/abs/2503.00084" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/arXiv-Paper-green"></a>
</p>

![GitHub Repo stars](https://img.shields.io/github/stars/FunAudioLLM/InspireMusic) Please support our community by starring it 感谢大家支持

| [**Highlight**](#highlights)
| [**Introduction**](#introduction)
| [**Installation**](#installation)
| [**Quick Start**](#quick-start)
| [**Tutorial**](https://github.com/FunAudioLLM/InspireMusic#tutorial)
| [**Models**](#models)

---
<a name="highlight"></a>
**InspireMusic** focuses on music generation, song generation, and audio generation.
- A unified toolkit designed for music, song, and audio generation.
- Music generation tasks with high audio quality. 
- Long-form music generation.

<a name="introduction"></a>
## Introduction
InspireMusic is a toolkit for music, song, and audio generation. It consists of an autoregressive transformer with a flow-matching based model. This toolkit is for users to generate music, song, and audio. InspireMusic can generate high-quality music in long-form with text-to-music and music continuation. InspireMusic incorporates audio tokenizers with autoregressive transformer and flow-matching modeling to generate music, song, and audio with text and music prompts. The toolkit currently supports music generation.

## InspireMusic
<p align="center"><table><tr><td style="text-align:center;"><img alt="Light" src="asset/InspireMusic.png" width="100%" /></tr><tr><td style="text-align:center;">
Figure 1: An overview of the InspireMusic. We introduce InspireMusic, a toolkit designed for music, song, audio generation capable of producing high-quality long-form music. InspireMusic consists of the following three key components. Audio Tokenizers convert the raw audio waveform into discrete audio tokens that can be efficiently processed and trained by the autoregressive transformer model. Audio waveform of lower sampling rate has converted to discrete tokens via a high bitrate compression audio tokenizer<a href="https://openreview.net/forum?id=yBlVlS2Fd9" target="_blank"><sup>[1]</sup></a>. Autoregressive Transformer model is based on Qwen2.5<a href="https://arxiv.org/abs/2412.15115" target="_blank"><sup>[2]</sup></a> as the backbone model and is trained using a next-token prediction approach on both text and audio tokens, enabling it to generate coherent and contextually relevant token sequences. The audio and text tokens are the inputs of an autoregressive model with the next token prediction to generate tokens. Super-Resolution Flow-Matching Model based on flow modeling method, maps the generated tokens to latent features with high-resolution fine-grained acoustic details<a href="https://arxiv.org/abs/2305.02765" target="_blank"><sup>[3]</sup></a> obtained from a higher sampling rate of audio to ensure the acoustic information flow connected with high fidelity through models. A vocoder then generates the final audio waveform from these enhanced latent features. InspireMusic supports a range of tasks including text-to-music, music continuation, music reconstruction and super resolution..
</td></tr></table></p>

<a name="installation"></a>
## Installation
### Clone
- Clone the repo
```
git clone --recursive https://github.com/FunAudioLLM/InspireMusic.git
# If you failed to clone submodule due to network failures, please run the following command until success
cd InspireMusic
git submodule update --recursive
# or you can download the third_party repo Matcha-TTS manually
cd third_party && git clone https://github.com/shivammehta25/Matcha-TTS.git
```

### Install from Source
InspireMusic requires Python>=3.8, PyTorch>=2.0.1, flash attention==2.6.2/2.6.3, CUDA>=11.8. You can install the dependencies with the following commands:

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
```
conda create -n inspiremusic python=3.8
conda activate inspiremusic
cd InspireMusic
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platforms.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# install flash attention to speedup training
pip install flash-attn --no-build-isolation
```

- Install within the package:
```
cd InspireMusic
# You can run to install the packages
python setup.py install
pip install flash-attn --no-build-isolation
```
We also recommend having `sox` or `ffmpeg` installed, either through your system or Anaconda:
```
# # Install sox
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel

# Install ffmpeg
# ubuntu
sudo apt-get install ffmpeg
# centos
sudo yum install ffmpeg
```

### Use Docker
Example command to build a docker image from Dockerfile provided.
```
docker build -t inspiremusic .
```
Example command to start the docker container in interactive mode.
```
docker run -ti --gpus all -v .:/workspace/InspireMusic inspiremusic
```

### Use Docker Compose
Example command to build a docker compose environment and docker image from the docker-compose.yml file.
```
docker compose up -d --build
```
Example command to attach to the docker container in interactive mode.
```
docker exec -ti inspire-music bash
```

<a name="quick-start"></a>
### Quick Start
an example command for music generation infer. 
```
cd InspireMusic
mkdir -p pretrained_models

# Download models
# ModelScope
git clone https://www.modelscope.cn/iic/InspireMusic-1.5B-Long.git pretrained_models/InspireMusic-1.5B-Long
# HuggingFace
git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git pretrained_models/InspireMusic-1.5B-Long

cd examples/music_generation
# run a quick inference example
sh infer_1.5b_long.sh
```
an example running script to run music generation task. 
```
cd InspireMusic/examples/music_generation/
sh run.sh
```

### Inference
#### Text-to-music Task
Example script for text-to-music task.
```
cd examples/music_generation
# with flow matching, use one-line command to get a quick try
python -m inspiremusic.cli.inference

# custom the config like the following one-line command
python -m inspiremusic.cli.inference --task text-to-music -m "InspireMusic-1.5B-Long" -g 0 -t "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance." -c intro -s 0.0 -e 30.0 -r "exp/inspiremusic" -o output -f wav 

# without flow matching, use one-line command to get a quick try
python -m inspiremusic.cli.inference --task text-to-music -g 0 -t "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance." --fast True
```

```
from inspiremusic.cli.inference import InspireMusicModel, env_variables
if __name__ == "__main__":
  env_variables()
  model = InspireMusicModel(model_name = "InspireMusic-Base")
  model.inference("text-to-music", "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance.")
```

#### Music Continuation Task
Example script for music continuation task.
```
cd examples/music_generation
# with flow matching
python -m inspiremusic.cli.inference --task continuation -g 0 -a audio_prompt.wav
# without flow matching
python -m inspiremusic.cli.inference --task continuation -g 0 -a audio_prompt.wav --fast True
```

```
from inspiremusic.cli.inference import InspireMusicModel
from inspiremusic.cli.inference import env_variables
if __name__ == "__main__":
  env_variables()
  model = InspireMusicModel(model_name = "InspireMusic-Base")
  # just use audio prompt
  model.inference("continuation", None, "audio_prompt.wav")
  # use both text prompt and audio prompt
  model.inference("continuation", "Continue to generate jazz music.", "audio_prompt.wav")
```
<a name="model"></a>
## Models
You may download our pretrained InspireMusic models for music generation.
```
# use git to download models，please make sure git lfs is installed.
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/InspireMusic.git pretrained_models/InspireMusic
```

### Available Models
Currently, we open source the music generation models support 24KHz mono and 48KHz stereo audio. 
The table below presents the links to the ModelScope and Huggingface model hub.

| Model name                           | Model Links                                                                                                                                                                                                                                                                                                                                   | Remarks                                                                                                  |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| InspireMusic-Base-24kHz              | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-Base-24kHz/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz)                                                                        | Pre-trained Music Generation Model, 24kHz mono, 30s                                                      |
| InspireMusic-Base                    | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-Base)                                                                                         | Pre-trained Music Generation Model, 48kHz, 30s                                                           |
| InspireMusic-1.5B-24kHz              | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz)                                                                        | Pre-trained Music Generation 1.5B Model, 24kHz mono, 30s                                                 |
| InspireMusic-1.5B                    | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B)                                                                                    | Pre-trained Music Generation 1.5B Model, 48kHz, 30s                                                      |
| InspireMusic-1.5B-Long               | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-Long/summary) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long)                                                                          | Pre-trained Music Generation 1.5B Model, 48kHz, support long-form music generation up to several minutes |
| InspireSong-1.5B                     | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                                                                                                                                                                          | Pre-trained Song Generation 1.5B Model, 48kHz stereo                                                     |
| InspireAudio-1.5B                    | [![model](https://img.shields.io/badge/ModelScope-Model-lightgrey.svg)]() [![model](https://img.shields.io/badge/HuggingFace-Model-lightgrey.svg)]()                                                                                                                                                                                          | Pre-trained Audio Generation 1.5B Model, 48kHz stereo                                                    |
| Wavtokenizer[<sup>[1]</sup>](https://openreview.net/forum?id=yBlVlS2Fd9) (75Hz) | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-Long/file/view/master?fileName=wavtokenizer%252Fmodel.pt) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long/tree/main/wavtokenizer)       | An extreme low bitrate audio tokenizer for music with one codebook at 24kHz audio.                       |
| Music_tokenizer (75Hz)               | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-24kHz/file/view/master?fileName=music_tokenizer%252Fmodel.pt) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz/tree/main/music_tokenizer) | A music tokenizer based on HifiCodec<sup>[3]</sup> at 24kHz audio.                                       |
| Music_tokenizer (150Hz)              | [![model](https://img.shields.io/badge/ModelScope-Model-green.svg)](https://modelscope.cn/models/iic/InspireMusic-1.5B-Long/file/view/master?fileName=music_tokenizer%252Fmodel.pt) [![model](https://img.shields.io/badge/HuggingFace-Model-green.svg)](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long/tree/main/music_tokenizer) | A music tokenizer based on HifiCodec<sup>[3]</sup> at 48kHz audio.                                       |

<a name="tutorial"></a>
## Basic Usage
At the moment, InspireMusic contains the training and inference codes for [music generation](https://github.com/FunAudioLLM/InspireMusic/tree/main/examples/music_generation). 

### Training
an example to train LLM model, support BF16/FP16 training. 
```
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=1024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    inspiremusic/bin/train.py \
    --train_engine "torch_ddp" \
    --config conf/inspiremusic.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model llm \
    --model_dir `pwd`/exp/music_generation/llm/ \
    --tensorboard_dir `pwd`/tensorboard/music_generation/llm/ \
    --ddp.dist_backend "nccl" \
    --num_workers 8 \
    --prefetch 100 \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer \
    --fp16
```
an example code to train flow matching model, does not support FP16 training.
```
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=1024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    inspiremusic/bin/train.py \
    --train_engine "torch_ddp" \
    --config conf/inspiremusic.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model flow \
    --model_dir `pwd`/exp/music_generation/flow/ \
    --tensorboard_dir `pwd`/tensorboard/music_generation/flow/ \
    --ddp.dist_backend "nccl" \
    --num_workers 8 \
    --prefetch 100 \
    --pin_memory \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
```

### Inference

An example script to quickly do model inference.
```
cd InspireMusic/examples/music_generation/
sh infer.sh
```
an example code to run inference with normal mode, i.e., with flow matching model for text-to-music and music continuation tasks.
```
pretrained_model_dir = "pretrained_models/InspireMusic/"
for task in 'text-to-music' 'continuation'; do
  python inspiremusic/bin/inference.py --task $task \
      --gpu 0 \
      --config conf/inspiremusic.yaml \
      --prompt_data data/test/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_model_dir/llm.pt \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --result_dir `pwd`/exp/inspiremusic/${task}_test \
      --chorus verse 
done
```
Here is an example code to run inference with fast mode, i.e., without flow matching model for text-to-music and music continuation tasks.
```
pretrained_model_dir = "pretrained_models/InspireMusic/"
for task in 'text-to-music' 'continuation'; do
  python inspiremusic/bin/inference.py --task $task \
      --gpu 0 \
      --config conf/inspiremusic.yaml \
      --prompt_data data/test/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_model_dir/llm.pt \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --result_dir `pwd`/exp/inspiremusic/${task}_test \
      --chorus verse \
      --fast 
done
```


## Roadmap
-  2024/12
  -  75Hz InspireMusic-Base model for music generation
    
-  2025/01
    -  Support to generate 48kHz
    -  75Hz InspireMusic-1.5B model for music generation
    -  75Hz InspireMusic-1.5B-Long model for long-form music generation

-  2025/02
    -  Release technical report

-  Future work
 - InspireAudio model for audio generation
 - InspireSong model for song generation
 - Support multilingual generation

## Citation
```
@inproceedings{InspireMusic2025,
      title={InspireMusic: Integrating Super Resolution and Large Language Model for High-Fidelity Long-Form Music Generation}, 
      author={Chong Zhang and Yukun Ma and Qian Chen and Wen Wang and Shengkui Zhao and Zexu Pan and Hao Wang and Chongjia Ni and Trung Hieu Nguyen and Kun Zhou and Yidi Jiang and Chaohong Tan and Zhifu Gao and Zhihao Du and Bin Ma},
      year={2025},
      eprint={2503.00084},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.00084}, 
}
```

## Acknowledgement
1. codes from CosyVoice, WavTokenizer，AcademiCodec，FunASR， FunCodec， Matcha-TTS，WeNet.

## Disclaimer
The content provided above is for research purposes only.
