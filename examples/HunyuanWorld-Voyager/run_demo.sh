#!/bin/bash

#    --i2v-dit-weight '/mnt/ceph1/libo/hf_models/HunyuanWorld-Voyager/Voyager/' \


export PT_HPU_LAZY_MODE=1
#export PT_HPU_LAZY_MODE=0
export PT_HPU_MAX_COMPOUND_OP_SIZE=256
#export PT_HPU_MAX_COMPOUND_OP_SIZE=65536
#export PT_HPU_MAX_COMPOUND_OP_SIZE=512
export PT_HPU_GPU_MIGRATION=1

export MODEL_BASE="/mnt/ceph1/libo/hf_models/HunyuanWorld-Voyager"

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



#multi card
#export ALLOW_RESIZE_FOR_SP=1 
#torchrun --nproc_per_node=8 \
#    sample_image2video.py \
#    --model HYVideo-T/2 \
#    --input-path "examples/case1" \
#    --prompt "An old-fashioned European village with thatched roofs on the houses." \
#    --i2v-stability \
#    --infer-steps 50 \
#    --flow-reverse \
#    --flow-shift 7.0 \
#    --seed 0 \
#    --embedded-cfg-scale 6.0 \
#    --save-path ./results \
#    --ulysses-degree 8 \
#    --ring-degree 1












