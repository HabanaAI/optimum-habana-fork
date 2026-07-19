cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD

if [[ "$OSTYPE" =~ ^darwin ]]; then
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

python3 infer/infer.py \
    --lrc-path infer/example/eg_en_full.lrc \
    --ref-audio-path infer/example/eg_en.mp3 \
    --audio-length 177 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 5