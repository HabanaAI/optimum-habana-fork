from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from stepvideo.config import parse_args
from stepvideo.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_world_group,
)
from stepvideo.utils import setup_seed

try:
    import habana_frameworks.torch.core as htcore
    from torch.hpu import set_device, device_count
except ModuleNotFoundError:
    pass


if __name__ == "__main__":
    args = parse_args()

    dist.init_process_group("hccl")
    init_distributed_environment(
        rank=dist.get_rank(),
        world_size=dist.get_world_size()
    )

    initialize_model_parallel(
        sequence_parallel_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        ulysses_degree=args.ulysses_degree,
    )

    local_rank = get_world_group().local_rank
    device = torch.device("hpu")

    setup_seed(args.seed)

    assert args.task in ["ti2v", "t2v"], "Please set the correct task, only ti2v and t2v are supported."

    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(dtype=torch.bfloat16, device="cpu")

    pipeline.transformer = pipeline.transformer.to(device)
    pipeline.setup_pipeline(args)

    prompt = args.prompt
    videos = pipeline(
        prompt=prompt, 
        first_image=args.first_image_path,
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width,
        num_inference_steps = args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=args.output_file_name or prompt[:50],
        motion_score=args.motion_score,
        seed=args.seed,
        task=args.task,
    )

    dist.destroy_process_group()
