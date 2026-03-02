#!/bin/bash

pip3 install torchaudio==2.9.0+cpu --user --no-deps --index-url https://download.pytorch.org/whl/cpu

pip3 install --user -r requirements.txt
apt update
apt install -y espeak-ng ffmpeg


