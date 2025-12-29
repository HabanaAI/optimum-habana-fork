# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from diffusers.utils import load_image
from diffusers import AutoPipelineForText2Image
import torch
import base64
from io import BytesIO
from PIL import Image
import os
import tempfile
import threading
import time
import re
import uuid

from typing import List, Union
from fastapi import Form, UploadFile

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import (
    ImgsGeneInputs,
    ImageOutputs,
)

logger = CustomLogger("opea_ImagesGenerations")
logflag = os.getenv("LOGFLAG", False)


pipe = None
args = None
initialization_lock = threading.Lock()
initialized = False


def initialize(
    model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
    device="cpu",
    token=None,
    bf16=True,
    use_hpu_graphs=False,
):
    global pipe, args, initialized
    with initialization_lock:
        if not initialized:
            # initialize model and tokenizer
            if os.getenv("MODEL", None):
                model_name_or_path = os.getenv("MODEL")
            kwargs = {}
            if bf16:
                kwargs["torch_dtype"] = torch.bfloat16
            if not token:
                token = os.getenv("HF_TOKEN")
            if device == "hpu":
                from optimum.habana.transformers.gaudi_configuration import GaudiConfig
                from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

                gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
                gaudi_config_kwargs["use_torch_autocast"] = True
                gaudi_config = GaudiConfig(**gaudi_config_kwargs)
                kwargs["use_habana"] = True
                kwargs["use_hpu_graphs"] = use_hpu_graphs
                kwargs["gaudi_config"] = gaudi_config
                kwargs["token"] = token

                if "qwen-image" in model_name_or_path.lower():
                    from optimum.habana.diffusers import GaudiQwenImagePipeline

                    adapt_transformers_to_gaudi()
                    pipe = GaudiQwenImagePipeline.from_pretrained(
                        model_name_or_path,
                        **kwargs,
                    )
                    logger.info("GaudiQwenImagePipeline loaded.")
                elif "stable-diffusion-3" in model_name_or_path.lower():
                    from optimum.habana.diffusers import GaudiStableDiffusion3Pipeline

                    adapt_transformers_to_gaudi()
                    pipe = GaudiStableDiffusion3Pipeline.from_pretrained(
                        model_name_or_path,
                        **kwargs,
                    )
                elif "stable-diffusion" in model_name_or_path.lower() or "flux" in model_name_or_path.lower():
                    from optimum.habana.diffusers import AutoPipelineForText2Image

                    adapt_transformers_to_gaudi()
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        model_name_or_path,
                        **kwargs,
                    )
                elif "z-image" in model_name_or_path.lower():
                    from optimum.habana.diffusers import GaudiStableDiffusionZImagePipeline

                    pipe = GaudiStableDiffusionZImagePipeline.from_pretrained(
                        model_name_or_path,
                        low_cpu_mem_usage=False,
                        **kwargs,
                    )

                else:
                    raise NotImplementedError(
                        "Only support qwen-image, stable-diffusion, stable-diffusion-xl, stable-diffusion-3, flux and z-image now, "
                        + f"model {model_name_or_path} not supported."
                    )
            elif device == "cpu":
                pipe = AutoPipelineForText2Image.from_pretrained(model_name_or_path, token=token, **kwargs)
                logger.info("AutoPipelineForText2Image loaded.")
            else:
                raise NotImplementedError(f"Only support cpu and hpu device now, device {device} not supported.")
            logger.info(f"device:{device} {model_name_or_path} model initialized.")
            initialized = True


@OpeaComponentRegistry.register("OPEA_IMAGES_GENERATIONS")
class OpeaImagesGenerations(OpeaComponent):
    """A specialized ImagesGenerations component derived from OpeaComponent for Stable Diffusion model .

    Attributes:
        model_name_or_path (str): The name of the Stable Diffusion model used.
        device (str): which device to use.
        token(str): Huggingface Token.
        bf16(bool): Is use bf16.
        use_hpu_graphs(bool): Is use hpu_graphs.
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: dict = None,
        seed=42,
        model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
        device="cpu",
        token=None,
        bf16=True,
        use_hpu_graphs=False,
    ):
        super().__init__(name, ServiceType.IMAGES_GENERATIONS.name.lower(), description, config)
        initialize(
            model_name_or_path=model_name_or_path, device=device, token=token, bf16=bf16, use_hpu_graphs=use_hpu_graphs
        )
        self.pipe = pipe
        self.seed = seed
        self.generator = torch.manual_seed(self.seed)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaImagesGenerations health check failed.")

    async def invoke(self, input: ImgsGeneInputs) -> ImageOutputs:
        """Invokes the ImagesGenerations service to generate Images for the provided input.

        Args:
            input (ImgsGeneInputs): The input in SD images  format.
        """

        start = time.time()

        prompt = input.prompt
        guidance_scale = self.config.get("guidance_scale", 4.0)
        true_cfg_scale = self.config.get("true_cfg_scale", 4.0)

        # Special step settings for GaudiStableDiffusionZImagePipeline (low=5, medium=9, high=20)
        is_zimage = hasattr(self.pipe, "__class__") and self.pipe.__class__.__name__ == "GaudiStableDiffusionZImagePipeline"
        if is_zimage:
            if input.quality is not None:
                if input.quality.lower() == "high":
                    num_inference_steps = 20
                elif input.quality.lower() == "medium":
                    num_inference_steps = 9
                elif input.quality.lower() == "low":
                    num_inference_steps = 5
                else:
                    num_inference_steps = self.config.get("num_inference_steps", 9)
            else:
                num_inference_steps = self.config.get("num_inference_steps", 9)
        else:
            # Step settings for other models (low=10, medium=25, high=50)
            if input.quality is not None:
                if input.quality.lower() == "high":
                    num_inference_steps = 50
                elif input.quality.lower() == "medium":
                    num_inference_steps = 25
                elif input.quality.lower() == "low":
                    num_inference_steps = 10
                else:
                    num_inference_steps = self.config.get("num_inference_steps", 25)
            else:
                num_inference_steps = self.config.get("num_inference_steps", 25)
        results_openai = []
        width = None
        height = None
        if input.size is not None:
            width_str, height_str = input.size.split("x")
            width = int(width_str) if width_str.isdigit() else None
            height = int(height_str) if height_str.isdigit() else None
        #clean_prompt = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', prompt)
        #prefix = clean_prompt[:2] if clean_prompt else "img"
        if logflag:
            logger.info(f"prompt: {prompt}, guidance_scale: {guidance_scale}, true_cfg_scale: {true_cfg_scale} quality: {input.quality} width: {width} height: {height} num_images_per_prompt:{int(input.n)}")

        images = pipe(prompt=prompt,
            generator=self.generator,
            true_cfg_scale=true_cfg_scale,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt = int(input.n)
            ).images

        #image_path = os.path.join(os.getcwd(), prefix)
        #os.makedirs(image_path, exist_ok=True)

        for i, image in enumerate(images):
            #save_path = os.path.join(image_path, f"image_{j}_{i+1}.png")
            #image.save(save_path)
            #with open(save_path, "rb") as f:
            #    bytes = f.read()
            #b64_str = base64.b64encode(bytes).decode()
            #we can directly convert image to bytes without saving to disk
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            results_openai.append({"b64_json": b64_str})

        return ImageOutputs(background="opaque", created=int(time.time()), data=results_openai, output_format="PNG", quality="high", size="0x0", usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_tokens_details": {"text_tokens": 0, "image_tokens": 0}},)

    def check_health(self) -> bool:
        """Checks the health of the ImagesGenerations service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            if self.pipe:
                return True
            else:
                return False
        except Exception as e:
            # Handle connection errors, timeouts, etc.
            logger.error(f"Health check failed: {e}")
        return False
