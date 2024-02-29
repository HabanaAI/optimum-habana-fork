HABANA_VISIBLE_MODULES="6,7" PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache,True,1024  \
python ../gaudi_spawn.py --world_size 2 --use_mpi --master_port 29501 train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path stabilityai/sdxl-vae \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --resolution 1024 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size 2 \
  --max_train_steps 16 \
  --learning_rate 1e-06 \
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
  --validation_epochs 48 \
  --checkpointing_steps 2500 \
  --logging_step 10 --max_train_samples 32 --mediapipe 2>&1 | tee log_2x_bs16.txt
#--mediapipe --max_train_samples 64