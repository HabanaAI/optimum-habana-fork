#!/bin/bash

export PT_HPU_LAZY_MODE=1
export PT_HPU_MAX_COMPOUND_OP_SIZE=256
export PT_HPU_GPU_MIGRATION=1

export MODEL_BASE="tencent/HunyuanWorld-Voyager"

#single card
python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "An old-fashioned European village with thatched roofs on the houses." \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --precision "bf16" \
    --vae-precision "bf16" \
    --text-encoder-precision "bf16" \
    --text-encoder-precision-2 "bf16" \
    --save-path ./results
