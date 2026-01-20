#!/bin/bash

export PT_HPU_LAZY_MODE=1
export USE_ZIMAGE_BUCKET=0
export FP32_SOFTMAX_VISION=0

#    --use_hpu_graphs \
python3 ./image_to_image_zimage_controlnet.py \
    --model_name_or_path 'Tongyi-MAI/Z-Image-Turbo/' \
    --controlnet_path 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors' \
    --pose_path 'pose.jpg' \
    --width 992 \
    --height 1728 \
    --controlnet_conditioning_scale 0.75 \
    --guidance_scale 0.0 \
    --num_inference_steps 9 \
    --seed 43  \
    --loop 5 \
    --prompts "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。" \
