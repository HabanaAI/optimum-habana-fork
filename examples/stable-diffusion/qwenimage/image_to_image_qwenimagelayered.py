import torch
import random
import numpy as np
import time as tm_perf

#from pipeline_qwen_image_layered import QwenImageLayeredPipelineGaudi
from optimum.habana.diffusers import GaudiQwenImageLayeredPipeline

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
#from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
import habana_frameworks.torch.gpu_migration

from optimum.habana.transformers.gaudi_configuration import GaudiConfig


from PIL import Image


def set_seed():
    seed = 5451
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class time_box_t():
    def __init__(self):
        self.t0=None

    def start(self):
        self.t0 = tm_perf.perf_counter()

    def show_time(self, desc):
        torch.cuda.synchronize()
        t1 = tm_perf.perf_counter()
        duration = t1-self.t0
        self.t0 = t1
        print(f'{desc} duration:{duration:.3f}s')




def main():
    set_seed()
    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = False
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": False,
        "gaudi_config": gaudi_config,
    }


    device = 'hpu'
    model_path = '/mnt/ceph1/libo/hf_models/Qwen-Image-Layered/'
    
    pipeline = QwenImageLayeredPipelineGaudi.from_pretrained(model_path, **kwargs)
    pipeline = pipeline.to(device, torch.bfloat16)
    pipeline.set_progress_bar_config(disable=None)
    
    image = Image.open("demo.png").convert("RGBA")
    inputs = {
        "image": image,
        "generator": torch.Generator(device=device).manual_seed(777),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "layers": 4,
        "resolution": 640,      # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
        "cfg_normalize": True,  # Whether enable cfg normalization.
        "use_en_prompt": True,  # Automatic caption language if user does not provide caption
    }
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
    
    for i, image in enumerate(output_image):
        image.save(f"{i}.png")



if "__main__" == __name__:
    main()

