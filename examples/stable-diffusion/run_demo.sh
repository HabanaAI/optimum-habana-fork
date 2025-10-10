#!/bin/bash

export PT_HPU_LAZY_MODE=1
#export PT_HPU_LAZY_MODE=0
#export PT_HPU_MAX_COMPOUND_OP_SIZE=256
export PT_HPU_MAX_COMPOUND_OP_SIZE=65536
#export PT_HPU_MAX_COMPOUND_OP_SIZE=512
export PT_HPU_GPU_MIGRATION=1

python3 ./text_to_image_kolors.py "一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”"

#python3 scripts/sample.py "a cat wears a hat"

# The image will be saved to "scripts/outputs/sample_test.jpg"
# The image will be saved to "scripts/outputs/sample_test.jpg"




