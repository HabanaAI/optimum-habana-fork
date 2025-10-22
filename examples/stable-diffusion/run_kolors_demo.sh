#!/bin/bash

export PT_HPU_LAZY_MODE=1
export PT_HPU_GPU_MIGRATION=1
export SHOW_KOLORS_PIPELINE_TIME=False

python3 ./text_to_image_kolors.py \
    --model_name_or_path 'Kwai-Kolors/Kolors' \
    --prompts "一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”" \
