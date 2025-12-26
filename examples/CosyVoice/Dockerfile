# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# HABANA environment for CosyVoice
FROM vault.habana.ai/gaudi-docker/1.21.4/ubuntu22.04/habanalabs/pytorch-installer-2.6.0

# Install git-lfs and other dependencies
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /CosyVoice

# Copy CosyVoice source code
COPY CosyVoice /CosyVoice

# Clone CosyVoice2-0.5B model from ModelScope
#RUN git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# Copy spk2info.pt to model directory
#RUN cp spk2info.pt pretrained_models/CosyVoice2-0.5B/

# Install Matcha-TTS (third party dependency)
RUN cd third_party/Matcha-TTS && \
    pip install --no-cache-dir .

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/CosyVoice:/usr/lib/habanalabs/

# Install Python dependencies
ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r requirements-hpu.txt && \
    $uvpip opea-comps==1.3

# Expose API server port (default port is 9370)
#EXPOSE 9370

# Default command: run API server
#CMD ["python", "api_server.py"]
