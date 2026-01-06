model=/data/Wan2.1-I2V-14B-480P-Diffusers
port=30088
# rank is used to set the number of cards
rank=4
out_dir=i2v_14b_480p_fp8_cp$rank

export HF_ENDPOINT=https://hf-mirror.com

prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." 
negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPU_LAZY_MODE=1 deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
    wan_i2v_quantization.py \
    --model_name_or_path $model \
    --prompt "$prompt" \
    --negative_prompts "$negative_prompt" \
    --image_path "i2v_input.jpg" \
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
    --quant_mode "quantize" \
    --quant_config "quantization/wan_i2v_480p/quantize_config.json"
