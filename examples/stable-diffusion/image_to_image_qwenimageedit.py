import os


os.environ["PT_HPU_LAZY_MODE"] = "1"
import argparse
import time

import torch
from PIL import Image

from optimum.habana.diffusers import GaudiQwenImageEditPlusPipeline
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen-Image-Edit-2509",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Change the picture to cartoon.",
        help="The prompt to guide the image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="The negative_prompt to guide the image generation.",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        nargs="*",
        default="",
        help="The image inputs to edit",
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

    adapt_transformers_to_gaudi()

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)

    pipeline = GaudiQwenImageEditPlusPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_habana=True,
        use_hpu_graphs=False,
        gaudi_config=gaudi_config,
    )

    image_list = []
    for path in args.images_path:
        image_list.append(Image.open(path))

    inputs = {
        "image": image_list,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "num_inference_steps": args.num_inference_steps,
    }

    with torch.inference_mode():
        # warmup
        output = pipeline(**inputs).images[0]

        t0 = time.time()
        output = pipeline(**inputs).images[0]
        t1 = time.time()
        print("Pipe time=", t1 - t0)
        out_path = "result_qwenimage_edit_2509.png"
        output.save(out_path)
        print("image saved at", out_path)


if __name__ == "__main__":
    main()
