#!/bin/bash

export PT_HPU_LAZY_MODE=1
export USE_ZIMAGE_BUCKET=0
export FP32_SOFTMAX_VISION=0

#    --use_hpu_graphs \
python3 ./image_to_image_zimage_controlnet_inpaint.py \
    --model_name_or_path '/mnt/ceph1/libo/hf_models/Z-Image-Turbo/' \
    --controlnet_path '/mnt/ceph1/libo/hf_models/Z-Image-Turbo-Fun-Controlnet-Union-2.1/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors' \
    --image_path '/mnt/ceph1/libo/zimage/inpaint/inpaint.jpg' \
    --mask_path '/mnt/ceph1/libo/zimage/inpaint/mask.jpg' \
    --pose_path '/mnt/ceph1/libo/zimage/inpaint/pose.jpg' \
    --width 992 \
    --height 1728 \
    --controlnet_conditioning_scale 0.75 \
    --guidance_scale 0.0 \
    --num_inference_steps 25 \
    --seed 43  \
    --loop 5 \
    --prompts "一位年轻女子站在阳光明媚的海岸线上，画面为全身竖构图，身体微微侧向右侧，左手自然下垂，右臂弯曲扶在腰间，她的手指清晰可见，站姿放松而略带羞涩。她身穿轻盈的白色连衣裙，裙摆在海风中轻轻飘动，布料半透、质感柔软。女子拥有一头鲜艳的及腰紫色长发，被海风吹起，在身侧轻盈飞舞，发间系着一个精致的黑色蝴蝶结，与发色形成对比。她面容清秀，眉目精致，肤色白皙细腻，表情温柔略显羞涩，微微低头，眼神静静望向远处的海平线，流露出甜美的青春气息与若有所思的神情。背景是辽阔无垠的海洋与蔚蓝天空，阳光从侧前方洒下，海面波光粼粼，泛着温暖的金色光晕，天空清澈明亮，云朵稀薄，整体色调清新唯美。" \
