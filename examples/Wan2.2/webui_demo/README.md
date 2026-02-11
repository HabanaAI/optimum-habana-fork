# Wan Series Intel Gaudi (HPU) Example on Web UI
This is an Intel Gaudi (HPU) Web UI example for Wan Series.

## Dependencies Installation

1. Install FFmpeg:

```bash
apt update && apt install ffmpeg
```

2. Install Python dependencies:

```bash
pip install -r ../requirements.txt
pip install -r ../requirements_animate.txt
pip install -r ../requirements_s2v.txt
```


## Demos

1. **Wan Animate 14b Web UI demo:**
Set the machine ip xxx to env no_proxy:
```bash
export no_proxy=xxx
```

Launch the server:

```bash
PT_HPU_GPU_MIGRATION=1 \
PT_HPU_LAZY_MODE=1 torchrun --nproc-per-node 8 generate_UI.py
--task animate_14B \
--ckpt_dir /mnt/ceph1/hf_models/Wan2.2-Animate-14B \
--refert_num 1  \
--base_seed 42 \
--ulysses_size 8
```

Run the web demo in another terminal:

```bash
python3 app.py
# Wait until the log prints: "* Running on local URL:  http://xxxx:xxxx"
```
