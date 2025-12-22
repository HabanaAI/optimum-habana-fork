#!/bin/bash

export PT_HPU_LAZY_MODE=1
export PT_HPU_GPU_MIGRATION=1
export USE_ZIMAGE_BUCKET=1
export FP32_SOFTMAX_VISION=0

python3 ./text_to_image_zimage.py \
    --model_name_or_path 'Tongyi-MAI/Z-Image-Turbo' \
    --width 512 \
    --height 512 \
    --guidance_scale 0.0 \
    --num_inference_steps 9 \
    --prompts "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights." \
