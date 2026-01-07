# 1.Qwen/Qwen-Image支持文生图
测试样例

PT_HPU_LAZY_MODE=1 python examples/stable-diffusion/qwenimage/text_to_image_qwenimage.py --model_name_or_path Qwen/Qwen-Image --prompt "A capybara wearing a suit holding a sign that reads Hello World." --num_inference_steps 20

参数设置：
--model_name_or_path 模型路径

--prompt 指导图像生成得prompt

--negative_prompt 指导图像生成的negative prompt， 默认为""

--height 生成图像的高,默认为1024

--width 生成图像的宽,默认为1024

--num_inference_steps diffusion的采样步数，步数越高图像越精细，耗时越长。默认为50

样例中包含：

1）使能optimum habana的优化

    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()

2） 模型pipeline启动：
    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)

    pipeline = GaudiQwenImagePipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_habana=True,
        use_hpu_graphs=False,
        gaudi_config=gaudi_config,
    )
    注意：

3）调用pipeline生成图像

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
        "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
    }
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

注意：

1> 当生图尺寸会发生变化时建议pipeline初始化时的参数使用use_hpu_graphs=False，这样可以避免OOM。如果生成图像只有一个固定尺寸，可以设置use_hpu_graphs=True，并配合环境变量QWENIMAGE_VAE_DECODE_BUCKETS=1 QWENIMAGE_TRANSFORMER_BUCKETS_STEP=1 来达到更好的性能。

2> 需要使用环境变量PT_HPU_LAZY_MODE=1

3> QwenImage的generator的manual_seed建议使用42，与官方例子保持一致。


# 2.Qwen/Qwen-Image-Edit 支持单图编辑
测试样例

PT_HPU_LAZY_MODE=1 python examples/stable-diffusion/qwenimage/image_to_image_qwenimageedit.py --model_name_or_path Qwen/Qwen-Image-Edit --prompt "Change to Cartoon style." --image_path /path/test.png --num_inference_steps 10

参数设置：

--model_name_or_path 模型路径

--prompt 指导图像生成得prompt

--negative_prompt 指导图像生成的negative prompt， 默认为""

--image_path 输入图像的路径

--num_inference_steps diffusion的采样步数，步数越高图像越精细，耗时越长。默认为50

样例中包含：
1）使能optimum habana的优化

    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()

2） 模型pipeline启动：

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)

    pipeline = GaudiQwenImageEditPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_habana=True,
        use_hpu_graphs=False,
        gaudi_config=gaudi_config,
    )

3）调用pipeline编辑图像

    inputs = {
        "image": image,
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

注意：

1> 需要使用环境变量PT_HPU_LAZY_MODE=1

2> Qwen/Qwen-Image-Edit的manual_seed建议使用0，与官方例子保持一致。


# 3.Qwen/Qwen-Image-Edit-2509 支持单图及多图编辑
测试样例

PT_HPU_LAZY_MODE=1 python examples/stable-diffusion/qwenimage/image_to_image_qwenimageeditplus.py --model_name_or_path Qwen/Qwen-Image-Edit-2509 --prompt "Change the two images into one cartoon picture." --images_path /path/img1.png /path/img2.png --num_inference_steps 10

参数设置：
--model_name_or_path 模型路径

--prompt 指导图像生成得prompt

--negative_prompt 指导图像生成的negative prompt， 默认为""

--images_path 输入图像的路径

--num_inference_steps diffusion的采样步数，步数越高图像越精细，耗时越长。默认为50

样例中包含：

1）使能optimum habana的优化

    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()

2） 模型pipeline启动：

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

3）调用pipeline编辑图像

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

注意：

1> 需要使用环境变量PT_HPU_LAZY_MODE=1

2> Qwen/Qwen-Image-Edit和Qwen/Qwen-Image-Edit-2509的manual_seed建议使用0，与官方例子保持一致。