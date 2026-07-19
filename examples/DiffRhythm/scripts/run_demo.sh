cd "$(dirname "$0")"
cd ../

export USE_DIFFRHYTHM_BUCKET=1
export PT_HPU_MAX_COMPOUND_OP_SIZE=512
export PT_HPU_LAZY_MODE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:$PWD

python3 infer/infer.py \
    --lrc-path infer/example/eg_cn_full.lrc \
    --ref-prompt "folk, acoustic guitar, harmonica, touching." \
    --audio-length 130 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 5
