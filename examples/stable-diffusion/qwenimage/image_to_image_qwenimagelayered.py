import argparse
import torch
import time

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()
from optimum.habana.diffusers import GaudiQwenImageLayeredPipeline
from optimum.habana.distributed import parallel_state
from optimum.habana.transformers.gaudi_configuration import GaudiConfig

from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen-Image-Layered",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="The image input path to edit",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense"
            " of slower inference."
        ),
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Determines how many ranks are divided into context parallel group.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Number of benchmark loops for generation.",
    )

    args = parser.parse_args()

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)

    if args.context_parallel_size > 1 and parallel_state.is_unitialized():
        if not torch.distributed.is_initialized():
            import deepspeed

            torch.distributed.init_process_group(backend="hccl")
            deepspeed.init_distributed(dist_backend="hccl")
        parallel_state.initialize_model_parallel(sequence_parallel_size=args.context_parallel_size, use_fp8=False)

    pipeline = GaudiQwenImageLayeredPipeline.from_pretrained(
        args.model_name_or_path,
        use_habana=True,
        use_hpu_graphs=False,
        gaudi_config=gaudi_config,
    )
    pipeline = pipeline.to("hpu", torch.bfloat16)
    image = Image.open(args.image_path).convert("RGBA")

    inputs = {
        "image": image,
        "generator": torch.Generator(device='cpu').manual_seed(777),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
        "layers": 4,
        "resolution": 640,      # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
        "cfg_normalize": True,  # Whether enable cfg normalization.
        "use_en_prompt": True,  # Automatic caption language if user does not provide caption
    }

    with torch.inference_mode():
        for idx in range(args.loop):
            t0 = time.time()
            output = pipeline(**inputs).images[0]
            torch.hpu.synchronize()
            t1 = time.time()
            duration = t1 - t0
            if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
                print("Qwen-Image-Layered Generation Latency in Loop #{:d}: {:.1f} sec".format(idx, duration))

        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            for i, image in enumerate(output):
                file_name = f"layered_{i}.png"
                image.save(file_name)
            print(f'Completed saving the images!')


if __name__ == "__main__":
    main()
