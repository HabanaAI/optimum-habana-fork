# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse

from typing import List, Union
from fastapi import Form, File, UploadFile, Depends, Request

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.images_edits.src.integrations.native import OpeaImagesEdits
from comps.cores.proto.api_protocol import (
    ImagesEditsInput,
    ImageOutputs,
)
logflag = os.getenv("LOGFLAG", False)
args = None
logger = CustomLogger("images_edits")
component_loader = None

async def resolve_images_edits_request(request: Request):
    form = await request.form()
    if logflag:
        for key in form.keys():
            values = form.getlist(key)
            for v in values:
                logger.info(f"{key}: {v}")

    images =[]
    if "image[]" in form:
        images += form.getlist("image[]")
    elif "image" in form:
        images += form.getlist("image")
    if logflag:
        logger.info(f"common_args images: {images}")
    common_args = {
        "image": images,
        "prompt": form.get("prompt", None),
        "background": form.get("background", None),
        "input_fidelity": form.get("input_fidelity", None),
        "mask": form.get("mask", None),
        "model": form.get("model", None),
        "n": int(form.get("n", 1)),
        "output_compression": int(form.get("output_compression", 100)),
        "output_format": form.get("output_format", None),
        "partial_images": int(form.get("partial_images", 0)),
        "quality": form.get("quality", None),
        "response_format": form.get("response_format", "url"),
        "size": form.get("size", None),
        "stream": form.get("stream", False),
        "user": form.get("user", None),
    }

    return ImagesEditsInput(**common_args)

@register_microservice(
    name="opea_service@images_edits",
    service_type=ServiceType.IMAGES_EDITS,
    endpoint="/v1/models",
    host="0.0.0.0",
    port=9390,
    methods=["GET"],
)
async def models():
    # TODO: need format here
    if os.getenv("MODEL", None):
        model_name_or_path = os.getenv("MODEL")
    else:
        model_name_or_path = "Qwen/Qwen-Image-Edit-2509"
    if logflag:
        logger.info(f"get model:{model_name_or_path}")
    model_info = {
        "object": "list",
        "data": [
            {
                "id": model_name_or_path,
                "object": "model",
                "created": None,
                "owned_by": model_name_or_path,
                "root": model_name_or_path,
                "parent": None,
                "max_model_len": None,
                "permission": [],
            }
        ],
    }

    return model_info

@register_microservice(
    name="opea_service@images_edits",
    service_type=ServiceType.IMAGES_EDITS,
    endpoint="/v1/images/edits",
    host="0.0.0.0",
    port=9390,
)

@register_statistics(names=["opea_service@images_edits"])
async def images_edits(input: ImagesEditsInput = Depends(resolve_images_edits_request)) -> ImageOutputs:
    start = time.time()
    results = await component_loader.invoke(input)
    statistics_dict["opea_service@images_edits"].append_latency(time.time() - start, None)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--use_hpu_graphs", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)

    args = parser.parse_args()
    if os.getenv("MODEL") is None:
        os.environ["MODEL"] = args.model_name_or_path
    images_edits_component_name = os.getenv("IMAGES_EDITS_COMPONENT_NAME", "OPEA_IMAGES_EDITS")
    # Register components
    try:
        # Initialize OpeaComponentLoader
        component_loader = OpeaComponentLoader(
            images_edits_component_name,
            description=f"OPEA IMAGES_EDITS Component: {images_edits_component_name}",
            config=args.__dict__,
            seed=args.seed,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            token=args.token,
            bf16=args.bf16,
            use_hpu_graphs=args.use_hpu_graphs,
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        exit(1)

    logger.info("images_edits server started.")
    opea_microservices["opea_service@images_edits"].start()
