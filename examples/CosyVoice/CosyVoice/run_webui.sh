#! /bin/bash

# This script copies the spk2info.pt file to the specified directory and then starts the web UI.
cp spk2info.pt /data/CosyVoice2-0.5B
python3 webui.py
