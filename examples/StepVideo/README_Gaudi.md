# Model Usage on Gaudi
We merge stepvideo ti2v and t2v as one repo here, check the following instructions to switch between different tasks.

## 1. Installation
```bash
cd optimum-habana/examples/StepVideo
pip install -e .
```

## 2. Run Inferences
Multi-cards Parallel Deployment

Stepvideo employed a decoupling strategy for the text encoder, VAE decoding, and DiT to optimize devices resource utilization by DiT. As a result, a dedicated device is needed to handle the API services for the text encoder's embeddings and VAE decoding.

Before start the inference, make sure to set:
```bash
export no_proxy=127.0.0.1
```

### 2.1 TI2V
Start vae and text encoder service:
```bash
python api/call_remote_server.py --model_dir /data/stepvideo-ti2v
```

Run DiT using multi-cards:
```bash
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node 2 run_parallel.py --model_dir /data/stepvideo-ti2v --vae_url "127.0.0.1" --caption_url "127.0.0.1" --ulysses_degree 2 --task 'ti2v' --prompt "笑起来" --first_image_path ./assets/demo.png --infer_steps 50 --cfg_scale 9.0 --time_shift 13.0 --motion_score 5.0 --output_file_name 'test' --name_suffix 'ti2v' --seed 42
```

### 2.2 T2V
Start vae and text encoder service
```bash
python api/call_remote_server.py --model_dir /data/stepvideo-t2v
```

Run DiT using multi-cards
```bash
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node 2 run_parallel.py --model_dir /data/stepvideo-t2v --vae_url "127.0.0.1" --caption_url "127.0.0.1" --ulysses_degree 2 --task 't2v' --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 50 --cfg_scale 9.0 --time_shift 13.0 --output_file_name 'test' --name_suffix 't2v' --num_frames 204 --seed 42
```

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--model_dir`       |   None    |   The model checkpoint for video generation    |
|     `--task`       |  ti2v    | Choose the model task between ti2v and t2v    |
|     `--prompt`     | “笑起来”  |      The text prompt for I2V generation      |
|    `--first_image_path`    |    ./assets/demo.png    |     The reference image path for I2V task.     |
|    `--infer_steps`     |    50     |     The number of steps for sampling      |
| `--cfg_scale` |    9.0    |    Embedded  Classifier free guidance scale       |
|     `--time_shift`     |    7.0    | Shift factor for flow matching schedulers. |
|     `--motion_score`   |    5.0  | Score to control the motion level of the video. |
|        `--seed`        |     None  |   The random seed for generating video, if None, we init a random seed    |
|  `--use-cpu-offload`   |   False   |    Use CPU offload for the model load to save more memory, necessary for high-res video generation    |
|     `--save-path`      | ./results |     Path to save the generated video      |
