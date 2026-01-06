model=/data/Wan2.2-T2V-A14B-Diffusers/
port=30088
rank=4
out_dir=14b_720p_cp$rank 

negative_prompt="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPU_LAZY_MODE=1 deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
    wan_t2v_quantization.py \
    --model_name_or_path $model \
    --negative_prompts "$negative_prompt" \
    --num_videos_per_prompt 1 \
    --use_habana \
    --seed 42 \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --output_type mp4 \
    --video_save_dir $out_dir \
    --dtype bf16 \
    --context_parallel_size $rank \
    --quant_mode "measure" \
    --quant_config "quantization/wan/measure_config.json" \
    --quant_config_2 "quantization/wan/measure_config_2.json"

