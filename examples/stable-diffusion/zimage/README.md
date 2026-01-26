# Z-image-Turbo 

## 测试样例

单卡：
```bash
PT_HPU_LAZY_MODE=1 \
USE_ZIMAGE_BUCKET=0 \
python examples/stable-diffusion/zimage/text_to_image_zimage.py \
    --pipeline_type "zimage" \
    --model_name_or_path 'Tongyi-MAI/Z-Image-Turbo/' \
    --width 512 \
    --height 512 \
    --guidance_scale 0.0 \
    --num_inference_steps 9 \
    --loop 3 \
    --prompts "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights." 
```

## 参数设置

    --model_name_or_path 模型路径

    --prompt 指导图像生成得prompt

    --pipeline_type 指定当前pipeine为 Z-image 

    --height 生成图像的高,默认为512

    --width 生成图像的宽,默认为512

    --num_inference_steps diffusion的采样步数，步数越高图像越精细，耗时越长。默认为9

样例中包含：

1）模型pipeline启动：

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = False
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }

    pipe = GaudiStableDiffusionZImagePipeline.from_pretrained(
        model_name_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        **kwargs,
    )

2）调用pipeline生成图像

    image = pipe(
        prompt=args.prompts,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,  # This actually results in 8 DiT forwards
        guidance_scale=args.guidance_scale,            # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

注意：

1）当生图尺寸会发生变化时建议pipeline初始化时的参数使用`use_hpu_graphs=False`，这样可以避免OOM。如果生成图像只有一个固定尺寸，可以设置`use_hpu_graphs=True`，并配合环境变量`USE_ZIMAGE_BUCKET=1`来达到更好的性能。

2）需要使用环境变量`PT_HPU_LAZY_MODE=1`

3）Zimage的generator的manual_seed建议使用42，与官方例子保持一致。

# Z-image-omni
## 测试样例

单卡：
```bash
PT_HPU_LAZY_MODE=1 \
USE_ZIMAGE_BUCKET=0 \
python examples/stable-diffusion/zimage/text_to_image_zimage.py \
    --pipeline_type "zimage_omni" \
    --model_name_or_path 'Z-a-o/Z-Image-Turbo/' \
    --width 1024 \
    --height 1024 \
    --guidance_scale 0.0 \
    --num_inference_steps 9 \
    --loop 3 \
    --prompts "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。" 
```

## 参数设置

    --model_name_or_path 模型路径

    --prompt 指导图像生成得prompt

    --pipeline_type 指定当前pipeine为 Z-image-omni

    --height 生成图像的高,默认为512

    --width 生成图像的宽,默认为512

    --num_inference_steps diffusion的采样步数，步数越高图像越精细，耗时越长。默认为9

样例中包含：

1）模型pipeline启动：

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = False
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }

    pipe = GaudiStableDiffusionZImageOmniPipeline.from_pretrained(
        model_name_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        **kwargs,
    )

3）调用pipeline生成图像

    image = pipe(
        prompt=args.prompts,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,  # This actually results in 8 DiT forwards
        guidance_scale=args.guidance_scale,            # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

注意：

由于Z-image-omni 模型并未发布当前代码仅为draft，无法运行！！！


#  Z-image controlnet 
## 测试样例

单卡：
```bash
PT_HPU_LAZY_MODE=1 \
python examples/stable-diffusion/zimage/image_to_image_zimage.py \
    --pipeline_type "controlnet" \
    --model_name_or_path 'Tongyi-MAI/Z-Image-Turbo' \
    --controlnet_path 'Z-Image-Turbo-Fun-Controlnet-Union-2.1/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors' \
    --pose_path 'pose.jpg' \
    --width 992 \
    --height 1728 \
    --controlnet_conditioning_scale 0.75 \
    --guidance_scale 0.0 \
    --num_inference_steps 9 \
    --seed 43  \
    --loop 5 \
    --prompt "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。" \
```

## 参数设置

    --model_name_or_path 模型路径

    --pipeline_type 指定当前pipeine为 Z-image controlnet

    --controlnet_path controlnet 位置

    --pose_path controlnet 需要的pose图片位置

    --prompt 指导图像生成的prompt

    --num_inference_steps diffusion的采样步数，步数越高图像越精细，耗时越长。默认为9

样例中包含：

1）controlnet 启动：

    controlnet = ZImageControlNetModel.from_single_file(
            args.controlnet_path,
            config='hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.1/config.json',
            torch_dtype = torch.bfloat16
    )

2）模型pipeline启动：

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }
    model_name_path = args.model_name_or_path

    pipe = GaudiStableDiffusionZImageControlNetPipeline.from_pretrained(
        model_name_path,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )

3）调用pipeline生成图片

    image = pipe(
        prompt=args.prompts,
        control_image=control_image,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

注意：

1）需要使用环境变量`PT_HPU_LAZY_MODE=1`

2）manual_seed建议使用43，与官方例子保持一致。

#  Z-image controlnet inpaint
## 测试样例

单卡：
```bash
PT_HPU_LAZY_MODE=1 \
python3 ./image_to_image_zimage.py \
    --pipeline_type "controlnet_inpaint" \
    --model_name_or_path 'Tongyi-MAI/Z-Image-Turbo' \
    --controlnet_path 'Z-Image-Turbo-Fun-Controlnet-Union-2.1/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors' \
    --image_path 'inpaint.jpg' \
    --mask_path 'mask.jpg' \
    --pose_path 'pose.jpg' \
    --width 992 \
    --height 1728 \
    --controlnet_conditioning_scale 0.75 \
    --guidance_scale 0.0 \
    --num_inference_steps 25 \
    --seed 43  \
    --loop 9 \
    --prompts "一位年轻女子站在阳光明媚的海岸线上，画面为全身竖构图，身体微微侧向右侧，左手自然下垂，右臂弯曲扶在腰间，她的手指清晰可见，站姿放松而略带羞涩。她身穿轻盈的白色连衣裙，裙摆在海风中轻轻飘动，布料半透、质感柔软。女子拥有一头鲜艳的及腰紫色长发，被海风吹起，在身侧轻盈飞舞，发间系着一个精致的黑色蝴蝶结，与发色形成对比。她面容清秀，眉目精致，肤色白皙细腻，表情温柔略显羞涩，微微低头，眼神静静望向远处的海平线，流露出甜美的青春气息与若有所思的神情。背景是辽阔无垠的海洋与蔚蓝天空，阳光从侧前方洒下，海面波光粼粼，泛着温暖的金色光晕，天空清澈明亮，云朵稀薄，整体色调清新唯美。" \
python examples/stable-diffusion/zimage/image_to_image_zimage.py \
```

## 参数设置

    --model_name_or_path 模型路径

    --pipeline_type 指定当前pipeine为 Z-image controlnet inpaint

    --controlnet_path controlnet 位置

    --image_path controlnet 需要的原始图片位置

    --mask_path controlnet 需要的掩码图片位置

    --pose_path controlnet 需要的pose图片位置

    --prompt 指导图像生成的prompt

    --num_inference_steps diffusion的采样步数，步数越高图像越精细，耗时越长。默认为25

样例中包含：

1）controlnet 启动：

    controlnet = ZImageControlNetModel.from_single_file(
            args.controlnet_path,
            config='hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.1/config.json',
            torch_dtype = torch.bfloat16
    )

2）模型pipeline启动：

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True
    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }
    model_name_path = args.model_name_or_path

    pipe = GaudiStableDiffusionZImageControlNetInpaintPipeline.from_pretrained(
            model_name_path, 
            controlnet = controlnet,
            torch_dtype=torch.bfloat16, 
            **kwargs
    )

3）调用pipeline生成图片

    image = pipe(
        prompt=args.prompts,
        image=image,
        mask_image=mask_image,
        control_image=control_image,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]


注意：

1）需要使用环境变量`PT_HPU_LAZY_MODE=1`

2）manual_seed建议使用43，与官方例子保持一致。
