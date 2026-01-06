export PT_HPU_SYNC_LAUNCH=1
export PT_HPU_LAZY_MODE=1

model="WanAI/Wan2.2-I2V-A14B-Diffusers"

port=30006
rank=8
bench_loop=3
out_dir=14b_i2v_cp$rank

deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
    image_to_video_generation.py \
    --model_name_or_path $model \
    --prompts "The cat removes the glasses from its eyes." \
    --image_path "./i2v_input.jpg" \
    --seed 42 \
    --use_habana \
    --height 1088 \
    --width 800 \
    --num_frames 81 \
    --bf16 \
    --fps 16 \
    --loop $bench_loop \
    --video_save_dir $out_dir \
    --context_parallel_size $rank
