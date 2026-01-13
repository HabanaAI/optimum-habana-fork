#!/bin/bash

export PT_HPU_LAZY_MODE=1
export USE_ZIMAGE_BUCKET=0
export FP32_SOFTMAX_VISION=0

python3 ./text_to_image_zimage_omni.py.py \
    --model_name_or_path 'Z-a-o/Z-Image-Turbo' \
    --width 1024 \
    --height 1024 \
    --guidance_scale 0.0 \
    --num_inference_steps 9 \
    --loop 3 \
    --prompts "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。" 
