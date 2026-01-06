export PT_HPU_SYNC_LAUNCH=1
export PT_HPU_LAZY_MODE=1

model="WanAI/Wan2.2-TI2V-5B-Diffusers"

port=30006
rank=8
bench_loop=3
out_dir=5b_t2v_cp$rank

deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
    text_to_video_generation.py \
    --model_name_or_path $model \
    --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --negative_prompts "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量 ，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 88 \
    --use_habana \
    --pipeline_type wan \
    --num_videos_per_prompt 1 \
    --height 704 \
    --width 1280 \
    --num_frames 121 \
    --loop $bench_loop \
    --num_inference_steps 50 \
    --dtype bf16 \
    --context_parallel_size $rank \
    --video_save_dir $out_dir
