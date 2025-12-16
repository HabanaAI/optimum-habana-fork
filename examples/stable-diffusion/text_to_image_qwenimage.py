import os


os.environ["PT_HPU_LAZY_MODE"] = "1"
import argparse
import time

import torch

from optimum.habana.diffusers import GaudiQwenImagePipeline
from optimum.habana.transformers.gaudi_configuration import GaudiConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen-Image",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A capybara wearing a suit holding a sign that reads Hello World.",
        help="The prompt to guide the image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="The negative_prompt to guide the image generation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width in pixels of the generated images (0=default from model config).",
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

    args = parser.parse_args()

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
    }

    pipeline = GaudiQwenImagePipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_habana=True,
        use_hpu_graphs=False,
        gaudi_config=gaudi_config,
    )

    inputs = {
        "prompt": args.prompt + positive_magic["en"],
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "generator": torch.Generator(device="cpu").manual_seed(42),
        "true_cfg_scale": 4.0,
        "num_inference_steps": args.num_inference_steps,
    }

    with torch.inference_mode():
        # warmup
        image = pipeline(**inputs).images[0]

        t0 = time.time()
        image = pipeline(**inputs).images[0]
        t1 = time.time()
        print("Pipe time=", t1 - t0)
        out_path = "result_qwenimage_result.png"
        image.save(out_path)
        print("image saved at", out_path)


if __name__ == "__main__":
    main()
