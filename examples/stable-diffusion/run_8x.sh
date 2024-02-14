PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache,True,1024  python ../gaudi_spawn.py --world_size 8 --use_mpi train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path stabilityai/sdxl-vae \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --resolution 1024 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size 16 \
  --max_train_steps 313 \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir sdxl-pokemon-model \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --dataloader_num_workers 8 \
  --bf16 \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a robotic cat with wings" \
  --validation_epochs 45 \
  --checkpointing_steps 313 \
  --cache_dir /root/software/data/pytorch/huggingface/sdxl 2>&1 | tee log_8x_bs16.txt
