# Use PyTorch 2.6 GPU base image with Python 3.11 and CUDA 12.1/12.4 on Ubuntu 22.04
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/FunAudioLLM/InspireMusic"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

# Set the working directory
WORKDIR /workspace/InspireMusic
# Copy the current directory contents into the container at /workspace/InspireMusic
git clone https://github.com/FunAudioLLM/InspireMusic.git

# inatall library dependencies
RUN apt-get update && apt-get install -y ffmpeg sox libsox-dev git && apt-get clean
RUN pip install -r requirements.txt

# install flash attention
RUN pip install flash-attn
