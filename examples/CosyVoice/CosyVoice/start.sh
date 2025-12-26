#!/bin/bash

cd /CosyVoice/
mkdir -p pretrained_models

if [[ -d "/models/CosyVoice2-0.5B" ]]; then
    echo "model path of CosyVoice existed"
else
    echo "model path of CosyVoice not existed!!!"
    echo "ERR: not able to load model because model path CosyVoice not existed"
    exit 1
fi
cp -r /models/CosyVoice2-0.5B pretrained_models/.
cp spk2info.pt pretrained_models/CosyVoice2-0.5B/.
mkdir -p /models/logs/
export PT_HPU_LAZY_MODE=1
echo "start CogVoice web service in 5 mins..."
exec python3 api_server.py --server-port 9370 2>&1 | tee /models/logs/Cogvoice.log 


