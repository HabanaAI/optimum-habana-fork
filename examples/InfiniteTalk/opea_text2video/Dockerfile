# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# HABANA environment
FROM vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0 AS hpu
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    mkdir -p /home/user/video && \
    chown -R user /home/user/

COPY src /home/user/text2video

RUN apt update && apt install -y ffmpeg
RUN cd /home/user && git clone https://github.com/HabanaAI/optimum-habana-fork.git -b aice/v1.22.0
RUN cd /home/user && git clone https://github.com/opea-project/GenAIComps.git
RUN chown -R user /home/user/text2video

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home/user/text2video:/usr/lib/habanalabs/:/home/user/optimum-habana-fork/examples/InfiniteTalk/infinitetalk/:/home/user/GenAIComps/

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r /home/user/text2video/requirements.txt && \
    $uvpip -r /home/user/text2video/requirements-infinitetalk.txt && \
    $uvpip 'git+https://github.com/HabanaAI/optimum-habana-fork.git@aice/v1.22.0'

USER user
WORKDIR /home/user/text2video
