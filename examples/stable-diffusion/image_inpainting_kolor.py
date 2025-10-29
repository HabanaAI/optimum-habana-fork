import torch
import time as tm_perf
import os, sys
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import StableDiffusionXLInpaintPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
#from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
import habana_frameworks.torch.gpu_migration
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.diffusers import GaudiStableDiffusionXLKolorsInpaintPipeline

class time_box_t():
    def __init__(self):
        self.t0=None

    def start(self):
        self.t0 = tm_perf.perf_counter()

    def show_time(self, desc):
        torch.hpu.synchronize()
        t1 = tm_perf.perf_counter()
        duration = t1-self.t0
        self.t0 = t1
        print(f'{desc} duration:{duration:.3f}s')

#def _pad_gaudi(
#        self,
#        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
#        max_length: Optional[int] = None,
#        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
#        pad_to_multiple_of: Optional[int] = None,
#        return_attention_mask: Optional[bool] = None,
#        padding_side: Optional[str] = "left",
#) -> dict:
#    """
#    Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
#    Args:
#        encoded_inputs:
#            Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
#        max_length: maximum length of the returned list and optionally padding length (see below).
#            Will truncate by taking into account the special tokens.
#        padding_strategy: PaddingStrategy to use for padding.
#            - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
#            - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
#            - PaddingStrategy.DO_NOT_PAD: Do not pad
#            The tokenizer padding sides are defined in self.padding_side:
#                - 'left': pads on the left of the sequences
#                - 'right': pads on the right of the sequences
#        pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
#            This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
#            `>= 7.5` (Volta).
#        return_attention_mask:
#            (optional) Set to False to avoid returning attention mask (default: set to model specifics)
#    """
#    # Load from model defaults
#    assert self.padding_side == "left"
#
#    required_input = encoded_inputs[self.model_input_names[0]]
#    seq_length = len(required_input)
#
#    if padding_strategy == PaddingStrategy.LONGEST:
#        max_length = len(required_input)
#
#    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
#        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
#
#    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
#
#    # Initialize attention mask if not present.
#    if "attention_mask" not in encoded_inputs:
#        encoded_inputs["attention_mask"] = [1] * seq_length
#
#    if "position_ids" not in encoded_inputs:
#        encoded_inputs["position_ids"] = list(range(seq_length))
#
#    if needs_to_be_padded:
#        difference = max_length - len(required_input)
#
#        if "attention_mask" in encoded_inputs:
#            encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
#        if "position_ids" in encoded_inputs:
#            encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
#        encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
#
#    return encoded_inputs
#
#
#setattr(ChatGLMTokenizer, "_pad", _pad_gaudi)

def infer(image_path, mask_path, prompt):
    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = False
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": gaudi_config,
    }

    ckpt_dir = '/mnt/ceph1/libo/hf_models/Kolors-Inpainting/'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()

    pipe = GaudiStableDiffusionXLKolorsInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            **kwargs,
    )
    
    pipe.to("hpu")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(603)
    basename = image_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    image = Image.open(image_path).convert('RGB')
    mask_image = Image.open(mask_path).convert('RGB')

    warmup = 3
    for i in range(warmup):
        result = pipe(
            prompt = prompt,
            image = image,
            mask_image = mask_image,
            height=1024,
            width=768,
            guidance_scale = 6.0,
            generator= generator,
            num_inference_steps= 5,
            negative_prompt = '残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量',
            num_images_per_prompt = 1,
            strength = 0.999
        ).images[0]
    torch.hpu.synchronize()

    time_box = time_box_t()
    time_box.start()
    result = pipe(
        prompt = prompt,
        image = image,
        mask_image = mask_image,
        height=1024,
        width=768,
        guidance_scale = 6.0,
        generator= generator,
        num_inference_steps= 25,
        negative_prompt = '残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量',
        num_images_per_prompt = 1,
        strength = 0.999
    ).images[0]
    time_box.show_time('pipelines')


    result.save(f'sample_inpainting_3.jpg')

if __name__ == '__main__':
    import fire
    fire.Fire(infer)
