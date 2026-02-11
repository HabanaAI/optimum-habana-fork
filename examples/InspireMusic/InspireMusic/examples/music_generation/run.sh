#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script just show an example to build your own music generation model.
# You may need to prepare your own dataset to fine-tune or train from scratch.
# Here take MusicCaps [1] dataset as an example.
# Download MusicCaps from: https://huggingface.co/datasets/google/MusicCaps

# Reference:
# 1. Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., Huang, Q., Jansen, A., Roberts, A., Tagliasacchi, M., Sharifi, M., Zeghidour, N., & Frank, C. (2023). MusicLM: Generating music from text. Google Research. https://doi.org/10.48550/arXiv.2301.11325

. ./path.sh || exit 1;

stage=1
stop_stage=5

model_name=InspireMusic-Base
pretrained_model_dir=../../pretrained_models/${model_name}
dataset_name=musiccaps

# data preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Download dataset and prepare wav.scp/text."
  # Here you may need to download MusicCaps dataset
  for x in ${dataset_name}_train ${dataset_name}_dev; do
    [ -d data/$x/ ] || mkdir -p data/$x/
  done
fi

export CUDA_VISIBLE_DEVICES="0"
# extract acoustic tokens
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract acoustic token, you should have prepared acoustic tokenizer model, wav.scp/text"
  for x in ${dataset_name}_dev ${dataset_name}_train; do
    echo "$x"
    tools/extract_acoustic_token.py --dir data/$x \
    --ckpt_path ${pretrained_model_dir}/music_tokenizer/model.pt \
    --config_path ${pretrained_model_dir}/music_tokenizer/config.json
  done
fi

# extract semantic tokens
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract semantic token, you should have prepared semantic tokenizer model, wav.scp/text"
  for x in ${dataset_name}_dev ${dataset_name}_train; do
    echo "$x"
    tools/extract_semantic_token.py --dir data/$x \
    --ckpt_path ${pretrained_model_dir}/wavtokenizer/model.pt \
    --config_path ${pretrained_model_dir}/wavtokenizer/config.yaml
  done
fi

# data packing
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2acoustic_token.pt/utt2semantic_token.pt"
  for x in ${dataset_name}_train ${dataset_name}_dev; do
    echo $x
    [ -d data/$x/parquet ] || mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 10000 \
      --num_processes 10 \
      --semantic_token_dir `pwd`/data/$x/ \
      --acoustic_token_dir `pwd`/data/$x/ \
      --des_dir `pwd`/data/$x/parquet \
      --src_dir `pwd`/data/$x/
  done
fi

# inference
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  test_set=${dataset_name}_dev
  echo "Run inference."
  expr_name="${test_set}"
  for task in 'text-to-music' 'continuation'; do
    [ -d `pwd`/exp/${model_name}/${task}_${expr_name} ] && rm -rf `pwd`/exp/${model_name}/${task}_${expr_name}
    echo `pwd`/exp/${model_name}/${task}_${expr_name}
    python inspiremusic/bin/inference.py --task $task \
        --gpu 0 \
        --config conf/inspiremusic.yaml \
        --prompt_data data/${test_set}/parquet/data.list \
        --flow_model $pretrained_model_dir/flow.pt \
        --llm_model $pretrained_model_dir/llm.pt \
        --music_tokenizer $pretrained_model_dir/music_tokenizer \
        --wavtokenizer $pretrained_model_dir/wavtokenizer \
        --result_dir `pwd`/exp/${model_name}/${task}_${expr_name}
  done
fi


# train llm and flow models fp16
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  job_id=1024
  dist_backend="nccl"
  num_workers=8
  prefetch=100
  train_engine=torch_ddp
  expr_name="InspireMusic-Base-musiccaps-ft"

  echo "Run model training. We support llm and flow traning."

  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat data/${dataset_name}_train/parquet/data.list > data/${dataset_name}_train.data.list
  cat data/${dataset_name}_dev/parquet/data.list > data/${dataset_name}_dev.data.list

  # train llm, support fp16 training
  model="llm"
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      inspiremusic/bin/train.py \
      --train_engine $train_engine \
      --config conf/inspiremusic.yaml \
      --train_data data/${dataset_name}_train.data.list \
      --cv_data data/${dataset_name}_dev.data.list \
      --model $model \
      --model_dir `pwd`/exp/${expr_name}/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/${expr_name}/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer \
      --fp16 \
      --checkpoint ../../pretrained_models/InspireMusic-Base/llm.pt

  # train flow matching model, only support fp32 training
  model="flow"
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      inspiremusic/bin/train.py \
      --train_engine $train_engine \
      --config conf/inspiremusic.yaml \
      --train_data data/${dataset_name}_train.data.list \
      --cv_data data/${dataset_name}_dev.data.list \
      --model $model \
      --model_dir `pwd`/exp/${expr_name}/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/${expr_name}/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
fi


