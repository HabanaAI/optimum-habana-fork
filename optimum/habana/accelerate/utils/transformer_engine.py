# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import torch
from peft.tuners.lora.layer import Linear as PEFTLinear
from optimum.habana.peft.layer import LoRALinear

try:
    import habana_frameworks.torch.hpex.experimental.transformer_engine as te
    from habana_frameworks.torch.hpex.experimental.transformer_engine.distributed import activation_checkpointing

    has_transformer_engine = True
except ImportError:
    has_transformer_engine = False


def is_fp8_available():
    return has_transformer_engine


def _convert_model(model, to_transformer_engine=True, _convert_linear=True):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
    """
    if not is_fp8_available():
        raise ImportError("Using `convert_model` requires transformer_engine to be installed.")
    for name, module in model.named_children():
        if type(module) == PEFTLinear and to_transformer_engine and _convert_linear:
            LoRALinear.replace_forward(module)
        if isinstance(module, torch.nn.Linear) and not type(module) == PEFTLinear and to_transformer_engine and _convert_linear:
            has_bias = module.bias is not None
            te_module = te.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype, skip_weight_param_allocation=True
            )
            te_module.weight = module.weight

            if has_bias:
                te_module.bias = module.bias

            setattr(model, name, te_module)
        elif isinstance(module, te.Linear) and not to_transformer_engine and _convert_linear:
            has_bias = module.bias is not None
            new_module = torch.nn.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            new_module.weight.copy_(module.weight)
            if has_bias:
                new_module.bias.copy_(module.bias)

            setattr(model, name, new_module)
        else:
            _convert_model(module, to_transformer_engine=to_transformer_engine, _convert_linear=_convert_linear)


def has_transformer_engine_layers(model):
    """
    Returns whether a given model has some `transformer_engine` layer or not.
    """
    if not is_fp8_available():
        raise ImportError("Using `has_transformer_engine_layers` requires transformer_engine to be installed.")
    for m in model.modules():
        if isinstance(m, (te.Linear)):
            return True
    return False

def convert_model(model):
    if not has_transformer_engine_layers(model):
        with torch.no_grad():
            _convert_model(model)
        model._converted_to_transformer_engine = True
    return model

def get_fp8_format(fp8_format):
    if fp8_format == "E5M2":
        return te.recipe.Format.E5M2
    elif fp8_format == "HYBRID":
        return te.recipe.Format.HYBRID
    else:
        raise ValueError

def get_fp8_recipe(fp8_config):
    fp8_config = dict(fp8_config) if fp8_config is not None else {}
    if "fp8_format" in fp8_config:
        fp8_config['fp8_format'] = get_fp8_format(fp8_config['fp8_format'])
    fp8_recipe_handler = te.recipe.DelayedScaling(**fp8_config)
    fp8_recipe_handler.backend = "TE"
    return fp8_recipe_handler

class FP8ContextWrapper:
    def __init__(self, ctx, fp8_recipe):
        self.ctx = ctx
        self.fp8_ctx = self.create_fp8_context(fp8_recipe)

    def __enter__(self):
        self.ctx.__enter__()
        self.fp8_ctx.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.fp8_ctx.__exit__(exc_type, exc_value, exc_traceback)
        self.ctx.__exit__(exc_type, exc_value, exc_traceback)

    @staticmethod
    def create_fp8_context(fp8_recipe):
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    # _gradient_checkpointing_func always takes the function to be recomputed as the first argument. The function
    # below wraps this first argument with transformer_engine activation_checkpointing context.
    @staticmethod
    def _gradient_checkpointing_wrap(func, *args, **kwargs):
        _args = list(args)
        _args[0] = activation_checkpointing()(_args[0])
        args = tuple(_args)

        return func(*args, **kwargs)

    @staticmethod
    def gradient_checkpointing_wrap(model):
        if hasattr(model, "gradient_checkpointing") and model.gradient_checkpointing:
            model._gradient_checkpointing_func = functools.partial(FP8ContextWrapper._gradient_checkpointing_wrap, model._gradient_checkpointing_func)
            return

        for module in model.modules():
            if hasattr(module, "gradient_checkpointing") and module.gradient_checkpointing:
                module._gradient_checkpointing_func = functools.partial(FP8ContextWrapper._gradient_checkpointing_wrap, module._gradient_checkpointing_func)
