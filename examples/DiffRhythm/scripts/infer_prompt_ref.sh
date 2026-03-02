cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD

if [[ "$OSTYPE" =~ ^darwin ]]; then
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

python3 infer/infer.py \
    --lrc-path infer/example/eg_cn_full.lrc \
    --ref-prompt "folk, acoustic guitar, harmonica, touching." \
    --audio-length 130 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 5
