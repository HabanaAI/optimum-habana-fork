model=/data/Wan2.1-I2V-14B-480P-Diffusers
port=30088
# rank is used to set the number of cards
rank=4
out_dir=i2v_14b_480p_cp$rank

export HF_ENDPOINT=https://hf-mirror.com

negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPU_LAZY_MODE=1 deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
    wan_i2v_quantization.py \
    --model_name_or_path $model \
    --negative_prompts "$negative_prompt" \
    --num_videos_per_prompt 1 \
    --use_habana \
    --seed 42 \
    --max_area 399360 \
    --num_frames 81 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --output_type mp4 \
    --video_save_dir $out_dir \
    --dtype bf16 \
    --context_parallel_size $rank \
    --quant_mode "measure" \
    --quant_config "quantization/wan_i2v_480p/measure_config.json"
