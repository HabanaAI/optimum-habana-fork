export PT_HPU_SYNC_LAUNCH=1
export PT_HPU_LAZY_MODE=1

model="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
port=30006
rank=8
bench_loop=2
out_dir=hunyuan15_i2v_cp$rank
prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
    image_to_video_generation.py \
    --model_name_or_path $model \
    --prompts "$prompt" \
    --image_path "./i2v_input.jpg" \
    --seed 1 \
    --use_habana \
    --num_frames 121 \
    --bf16 \
    --fps 16 \
    --loop $bench_loop \
    --video_save_dir $out_dir \
    --context_parallel_size $rank
