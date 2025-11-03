#!/bin/bash

export SHOW_KOLORS_PIPELINE_TIME=False
export PT_HPU_LAZY_MODE=1
export PT_HPU_GPU_MIGRATION=1

python3 image_inpainting_kolor.py \
    --model_name_or_path 'Kwai-Kolors/Kolors-Inpainting' \
    --image_path ./inpainting/asset/3.png \
    --mask_path ./inpainting/asset/3_mask.png \
    --prompts "穿着美少女战士的衣服，一件类似于水手服风格的衣服，包括一个白色紧身上衣，前胸搭配一个大大的红色蝴蝶结。衣服的领子部分呈蓝色，并且有白色条纹。她还穿着一条蓝色百褶裙，超高清，辛烷渲染，高级质感，32k，高分辨率，最好的质量，超级细节，景深" \
