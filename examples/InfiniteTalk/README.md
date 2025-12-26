# InfiniteTalk
This is an Intel Gaudi (HPU) example for the [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) model.

## Dependencies Installation

1. Install FFmpeg:

```bash
apt update && apt install ffmpeg
```

2. Install Python dependencies:

```bash
pip install -r requirements-hpu.txt
```

## Model Preparation

Download model weights:

```bash
cd infinitetalk
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
```

## Demos

1. **Run on a single HPU:**

```bash
PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 python generate_infinitetalk.py \
    --ckpt_dir  ./weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir ./weights/chinese-wav2vec2-base \
    --infinitetalk_dir ./weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res
```

2. **Run on 4 HPUs (Sequence Parallel enabled):**

```bash
PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node=4 --standalone generate_infinitetalk.py \
    --ckpt_dir  ./weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir ./weights/chinese-wav2vec2-base \
    --infinitetalk_dir ./weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res_multihpu \
    --ulysses_size=4
```

3. **Run on a single HPU (Multi-Person):**

```bash
PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 python generate_infinitetalk.py \
    --ckpt_dir  ./weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir ./weights/chinese-wav2vec2-base \
    --infinitetalk_dir ./weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res_multiperson
```

4. **Run on 4 HPUs (Sequence Parallel enabled, Multi-Person):**

```bash
PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node=4 --standalone generate_infinitetalk.py \
    --ckpt_dir  ./weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir ./weights/chinese-wav2vec2-base \
    --infinitetalk_dir ./weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --input_json examples/multi_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res_multihpu_multiperson \
    --ulysses_size=4
```

5. **Web UI demo:**

Launch the server:

```bash
PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node=8 --standalone generate_infinitetalk_UI.py \
    --ckpt_dir  ./weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir ./weights/chinese-wav2vec2-base \
    --infinitetalk_dir ./weights/InfiniteTalk/single/infinitetalk.safetensors \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --save_file infinitetalk_res_multihpu \
    --ulysses_size=8
```

Run the web demo in another terminal:

```bash
python app.py --cp 8
# Wait until the log prints: "* Running on local URL:  http://xxxx:xxxx"
```

## Configuration

| Parameter         | Description                                   | Values / Default             | Notes                                                           |
|-------------------|-----------------------------------------------|------------------------------|-----------------------------------------------------------------|
| `--mode`          | Video generation mode                         | `streaming` (default)<br>`clip` | `streaming` supports continuous long-video generation.<br>`clip` outputs a single chunk. |
| `--size`          | Output video resolution                       | `infinitetalk-480` (default)<br>`infinitetalk-720` | Supports 480p and 720p.                   |
| `--sample_steps`  | Sampling steps                                | `40`                         | `40` recommended for image-to-video and `50` for text-to-video. |
| `--max_frame_num` | Maximum number of frames to generate          | `1000` (≈ 40s at 25 FPS)     |                                                                 |
| `--frame_num`     | Number of frames generated per clip           | `81`                         | Must be `4n + 1`.                                               |
| `--motion_frame`  | Driven frame length for long-video generation | `9`                          |                                                                 |
| `--base_seed`     | Random seed                                   | `42`                         |                                                                 |
| `--offload_model` | Offload model to CPU after each forward pass  | `True`                       | `False` is recommended for HPU.                                 |
| `--ulysses_size`  | Ulysses (sequence parallel) size              | `1` (default)                | Sequence-parallel degree.                                       |
