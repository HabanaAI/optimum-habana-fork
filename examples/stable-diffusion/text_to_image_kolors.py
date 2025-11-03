import os, torch
import argparse
import time
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler
from optimum.habana.diffusers import GaudiStableDiffusionKolorsPipeline
from optimum.habana.transformers.gaudi_configuration import GaudiConfig

from kolors.models.modeling_chatglm import ChatGLMModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Kolors",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="An image of a squirrel in Picasso style",
        help="The prompt or prompts to guide the image generation.",
    )
    args = parser.parse_args()

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": gaudi_config,
    }
    kwargs["force_zeros_for_empty_prompt"]=False

    ckpt_dir = args.model_name_or_path
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.bfloat16).to(torch.bfloat16)
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).bfloat16()
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).bfloat16()

    pipe = GaudiStableDiffusionKolorsPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            **kwargs,)
    pipe = pipe.to("hpu")

    warmup = 2
    for i in range(warmup):
        pipe(
            prompt=args.prompts,
            height=1024,
            width=1024,
            num_inference_steps=3,
            guidance_scale=5.0,
            num_images_per_prompt=1,
            generator= torch.Generator(pipe.device).manual_seed(878))
    torch.hpu.synchronize()

    iter_time = 0
    start_time = time.perf_counter()
    for i in range(5):
        image = pipe(
            prompt=args.prompts,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=5.0,
            num_images_per_prompt=1,
            is_profiler = False,
            generator= torch.Generator(pipe.device).manual_seed(5544)).images[0]
    torch.hpu.synchronize()
    iter_time += time.perf_counter() - start_time

    iter_time = iter_time / 5
    print(f'Kolors pipeline duration:{iter_time:.3f}s')

    image.save(f'piaocong.jpg')

if __name__ == '__main__':
    main()
