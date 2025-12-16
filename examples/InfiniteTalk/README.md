# InfiniteTalk
this is (MeiGen-AI/InfiniteTalk)[https://github.com/MeiGen-AI/InfiniteTalk] model example on Intel Gaudi.

1. prepare envs.
```bash
apt update && apt install ffmpeg
pip install -r requirements-hpu.txt
```
2. download weights.
```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk

```
3. run demo script:
single device:
```bash
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 python generate_infinitetalk.py \
    --ckpt_dir  /mnt/new_disk/models/Wan2.1-I2V-14B-480P \
    --wav2vec_dir '/mnt/new_disk/models/chinese-wav2vec2-base' \
    --infinitetalk_dir  /mnt/new_disk/models/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res
```

multi device with sequence parallel:
```bash
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node=4 --standalone generate_infinitetalk.py \
    --ckpt_dir  /mnt/new_disk/models/Wan2.1-I2V-14B-480P \
    --wav2vec_dir '/mnt/new_disk/models/chinese-wav2vec2-base' \
    --infinitetalk_dir  /mnt/new_disk/models/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res_multihpu \
    --ulysses_size=4
```

4. run web_ui demo:
multi device with sequence parallel
```bash
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node=8 --standalone generate_infinitetalk_UI.py \
    --ckpt_dir  /mnt/new_disk/models/Wan2.1-I2V-14B-480P \
    --wav2vec_dir '/mnt/new_disk/models/chinese-wav2vec2-base' \
    --infinitetalk_dir  /mnt/new_disk/models/InfiniteTalk/single/infinitetalk.safetensors \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res_multihpu \
    --ulysses_size=8
```
run web_demo in another terminal
```bash
python app.py --cp 8
# wait until log print "* Running on local URL:  http:xxxx:xxxx"
```

