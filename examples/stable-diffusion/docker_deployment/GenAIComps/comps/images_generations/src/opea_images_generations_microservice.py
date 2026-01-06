# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse

from typing import List, Union
from fastapi import Form, UploadFile

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    SDImg2ImgInputs,
    SDOutputs,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.images_generations.src.integrations.native import OpeaImagesGenerations
from comps.cores.proto.api_protocol import (
    ImgsGeneInputs,
    ImageOutputs,
)
logflag = os.getenv("LOGFLAG", False)
args = None
logger = CustomLogger("images_generations")
component_loader = None

@register_microservice(
    name="opea_service@images_generations",
    service_type=ServiceType.IMAGES_GENERATIONS,
    endpoint="/v1/models",
    host="0.0.0.0",
    port=9391,
    methods=["GET"],
)
async def models():
    # TODO: need format here
    if os.getenv("MODEL", None):
        model_name_or_path = os.getenv("MODEL")
    else:
        model_name_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
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
    name="opea_service@images_generations",
    service_type=ServiceType.IMAGES_GENERATIONS,
    endpoint="/v1/images/generations",
    host="0.0.0.0",
    port=9391,
    input_datatype=ImgsGeneInputs,
    output_datatype=ImageOutputs,
)

@register_statistics(names=["opea_service@images_generations"])
async def images_generations(input: ImgsGeneInputs) -> ImageOutputs:
    start = time.time()
    results = await component_loader.invoke(input)
    statistics_dict["opea_service@images_generations"].append_latency(time.time() - start, None)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0")
    parser.add_argument("--use_hpu_graphs", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)

    args = parser.parse_args()
    if os.getenv("MODEL") is None:
        os.environ["MODEL"] = args.model_name_or_path
    images_generations_component_name = os.getenv("IMAGES_GENERATIONS_COMPONENT_NAME", "OPEA_IMAGES_GENERATIONS")
    # Register components
    try:
        # Initialize OpeaComponentLoader
        component_loader = OpeaComponentLoader(
            images_generations_component_name,
            description=f"OPEA IMAGES_GENERATIONS Component: {images_generations_component_name}",
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

    logger.info("images_generations server started.")
    opea_microservices["opea_service@images_generations"].start()
