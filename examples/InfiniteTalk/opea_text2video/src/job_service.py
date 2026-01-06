# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import re
import wan
import subprocess
import torch
import random
import argparse
import logging
import os
import sys
import json
import time
import warnings
import librosa
import numpy as np
import torch.distributed as dist
import soundfile as sf
import pyloudnorm as pyln
import fcntl
import imageio

from tqdm import tqdm
from einops import rearrange
from kokoro import KPipeline
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from wan.utils.segvideo import shot_detect
from wan.utils.multitalk_utils import save_video_ffmpeg, cache_video
from wan.utils.utils import str2bool, is_video, split_wav_librosa
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS


warnings.filterwarnings("ignore")


def save_video_with_logo(gen_video_samples, save_path, vocal_audio_list, fps=25, quality=5, high_quality_save=False):

    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            writer.append_data(frame)
        writer.close()
    save_path_tmp = save_path + "-temp.mp4"

    if high_quality_save:
        cache_video(
            tensor=gen_video_samples.unsqueeze(0),
            save_file=save_path_tmp,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
    else:
        video_audio = (gen_video_samples+1)/2  # C T H W
        video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
        video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
        save_video(video_audio, save_path_tmp, fps=fps, quality=quality)

    # crop audio according to video length
    C, T, H, W = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = save_path + "-cropaudio.wav"
    final_command = [
        "ffmpeg",
        "-i",
        vocal_audio_list[0],
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)
    logo_w = 1280
    logo_h = 720
    if W / H > logo_w / logo_h:
        ratio = H / logo_h
        resized_logo_h = H
        resized_logo_w = logo_w * ratio
        pad_w = (W - resized_logo_w) / 2
        pad_h = 0
    else:
        ratio = W / logo_w
        resized_logo_w = W
        resized_logo_h = logo_h * ratio
        pad_h = (H - resized_logo_h) / 2
        pad_w = 0
    save_path = save_path + ".mp4"
    if high_quality_save:
        final_command = [
            "ffmpeg",
            "-y",
            "-i", save_path_tmp,
            "-i", save_path_crop_audio,
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", "veryslow",
            "-c:a", "aac",
            "-shortest",
            save_path,
        ]
        subprocess.run(final_command, check=True)
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)
    else:
        final_command = [
            "ffmpeg",
            "-y",
            "-i",
            save_path_tmp,
            "-i",
            save_path_crop_audio,
            "-i",
            "/home/user/video/intel_logo.mp4",
            "-filter_complex",
            f"[2:v]scale=w={int(resized_logo_w)}:h={int(resized_logo_h)},"
            f"setdar=0x0,pad={W}:{H}:{int(pad_w)}:{int(pad_h)}:black[2v],[0:v][1:a][2v][2:a]concat=n=2:v=1:a=1[v][a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            "-vsync",
            "passthrough",
            save_path,
        ]
        subprocess.run(final_command, check=True)
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)


def find_max_matching_frame(max_value: int, default_value: int) -> int:
    """
    Finds the largest integer less than or equal to max_value
    that can be expressed in the form 4*n + 1.

    Args:
        max_value: The upper bound for the search.

    Returns:
        The largest number matching the pattern, or None if no such
        number exists within the given limit (e.g., if max_value < 1).
    """
    # The smallest number of the form 4*n + 1 (for n>=0) is 1.
    if max_value < 1:
        return default_value

    # Start from max_value and check downwards.
    for number in range(max_value, 0, -1):
        # A number is of the form 4*n + 1 if its remainder when divided by 4 is 1.
        if number % 4 == 1:
            return number

    return default_value  # Should not be reached if max_value >= 1


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, 99999999)
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], (f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument("--task", type=str, default="infinitetalk-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument("--size", type=str, default="infinitetalk-480", choices=list(SIZE_CONFIGS.keys()), help="The buckget size of the generated video. The aspect ratio of the output video will follow that of the input image.",)
    parser.add_argument("--max_frame_num", type=int, default=100000, help="The max frame lenght of the generated video.")
    parser.add_argument("--ckpt_dir", type=str, default="/hf/Wan2.1-I2V-14B-480P", help="The path to the Wan checkpoint directory.")
    parser.add_argument("--infinitetalk_dir", type=str, default="/hf/InfiniteTalk/single/infinitetalk.safetensors", help="The path to the InfiniteTalk checkpoint directory.")
    parser.add_argument("--wav2vec_dir", type=str, default="/hf/chinese-wav2vec2-base", help="The path to the wav2vec checkpoint directory.")
    parser.add_argument("--quant_dir", type=str, default=None, help="The path to the Wan quant checkpoint directory.")
    parser.add_argument("--dit_path", type=str, default=None, help="The path to the Wan checkpoint directory.")
    parser.add_argument("--base_seed", type=int, default=42, help="The seed to use for generating the image or video.")
    parser.add_argument("--lora_dir", type=str, nargs="+", default=None, help="The paths to the LoRA checkpoint files.")
    parser.add_argument("--lora_scale", type=float, nargs="+", default=[1.2], help="Controls how much to influence the outputs with the LoRA parameters. Accepts multiple float values.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
    parser.add_argument("--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
    parser.add_argument("--mode", type=str, default="clip", choices=["clip", "streaming"], help="clip: generate one video chunk, streaming: long video generation")
    parser.add_argument("--audio_mode", type=str, default="localfile", choices=["localfile", "tts"], help="localfile: audio from local wav file, tts: audio from TTS")
    parser.add_argument("--motion_frame", type=int, default=9, help="Driven frame length used in the mode of long video genration.")
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None, required=False, help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required")
    parser.add_argument("--use_teacache", action="store_true", default=False, help="Enable teacache for video generation.")
    parser.add_argument("--teacache_thresh", type=float, default=0.2, help="Threshold for teacache.")
    parser.add_argument("--use_apg", action="store_true", default=False, help="Enable adaptive projected guidance for video generation (APG).")
    parser.add_argument("--apg_momentum", type=float, default=-0.75, help="Momentum used in adaptive projected guidance (APG).")
    parser.add_argument("--apg_norm_threshold", type=float, default=55, help="Norm threshold used in adaptive projected guidance (APG).")
    parser.add_argument("--color_correction_strength", type=float, default=1.0, help="strength for color correction [0.0 -- 1.0].")
    parser.add_argument("--scene_seg", action="store_true", default=False, help="Enable scene segmentation for input video.")
    parser.add_argument("--quant", type=str, default=None, help="Quantization type, must be 'int8' or 'fp8'.")
    parser.add_argument("--video_dir", type=str, default="/home/user/video", help="Video output directory.")
    parser.add_argument("--sep", type=str, default="$###$", help="Video output directory.")

    args = parser.parse_args()
    _validate_args(args)
    return args


def custom_init(device, wav2vec):
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager").to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    if not (left_path == "None" or right_path == "None"):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path == "None":
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path == "None":
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type == "para":
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type == "add":
        new_human_speech1 = np.concatenate(
            [human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]
        )
        new_human_speech2 = np.concatenate(
            [np.zeros(human_speech_array1.shape[0]), human_speech_array2[: human_speech_array2.shape[0]]]
        )
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device="cpu"):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split("/")[-1].split(".")[0] + ".wav"
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array


def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array


def process_tts_single(text, save_dir, voice1):
    s1_sentences = []

    pipeline = KPipeline(lang_code="a", repo_id="weights/Kokoro-82M")

    voice_tensor = torch.load(voice1, weights_only=True)
    generator = pipeline(
        text,
        voice=voice_tensor,  # <= change voice here
        speed=1,
        split_pattern=r"\n+",
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = torch.concat(audios, dim=0)
    s1_sentences.append(audios)
    s1_sentences = torch.concat(s1_sentences, dim=0)
    save_path1 = f"{save_dir}/s1.wav"
    sf.write(save_path1, s1_sentences, 24000)  # save each audio file
    s1, _ = librosa.load(save_path1, sr=16000)
    return s1, save_path1


def process_tts_multi(text, save_dir, voice1, voice2):
    pattern = r"\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    s1_sentences = []
    s2_sentences = []

    pipeline = KPipeline(lang_code="a", repo_id="weights/Kokoro-82M")
    for idx, (speaker, content) in enumerate(matches):
        if speaker == "1":
            voice_tensor = torch.load(voice1, weights_only=True)
            generator = pipeline(
                content,
                voice=voice_tensor,  # <= change voice here
                speed=1,
                split_pattern=r"\n+",
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s1_sentences.append(audios)
            s2_sentences.append(torch.zeros_like(audios))
        elif speaker == "2":
            voice_tensor = torch.load(voice2, weights_only=True)
            generator = pipeline(
                content,
                voice=voice_tensor,  # <= change voice here
                speed=1,
                split_pattern=r"\n+",
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s2_sentences.append(audios)
            s1_sentences.append(torch.zeros_like(audios))

    s1_sentences = torch.concat(s1_sentences, dim=0)
    s2_sentences = torch.concat(s2_sentences, dim=0)
    sum_sentences = s1_sentences + s2_sentences
    save_path1 = f"{save_dir}/s1.wav"
    save_path2 = f"{save_dir}/s2.wav"
    save_path_sum = f"{save_dir}/sum.wav"
    sf.write(save_path1, s1_sentences, 24000)  # save each audio file
    sf.write(save_path2, s2_sentences, 24000)
    sf.write(save_path_sum, sum_sentences, 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    # sum, _ = librosa.load(save_path_sum, sr=16000)
    return s1, s2, save_path_sum


def update_job(job_processed, args):
    # If a job was processed, rewrite the entire job file
    job_file = os.path.join(args.video_dir, "job.txt")
    sep = args.sep
    if job_processed:
        with open(job_file, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Re-read the file to get the latest content before writing
                f.seek(0)
                lines_before_write = [line.strip() for line in f if line.strip()]

                # Find the job by ID and update it
                job_id_to_update = job_processed[0]
                found = False
                for i, line in enumerate(lines_before_write):
                    if line.startswith(job_id_to_update + sep):
                        lines_before_write[i] = sep.join(map(str, job_processed))
                        found = True
                        break

                # If the job was somehow removed from the file, add the new status at the end
                if not found:
                    lines_before_write.append(sep.join(map(str, job_processed)))

                # Write the updated content back to the file
                f.seek(0)
                f.truncate()
                for line in lines_before_write:
                    f.write(line + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), (f"t5_fsdp and dit_fsdp are not supported in non-distributed environments.")
        assert not (args.ulysses_size > 1 or args.ring_size > 1), (f"context parallel are not supported in non-distributed environments.")

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, (f"The number of ulysses_size and ring_size should be equal to the world size.")
        assert args.ulysses_size * args.ring_size <= 8, (f"Currently, sequence parallel degree should be no larger than 8.")  # TODO: remove this limit in the future
        from wan.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    assert args.task == "infinitetalk-14B", "You should choose infinitetalk in args.task."

    logging.info("Creating infinitetalk pipeline.")
    wan_i2v = wan.InfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        quant_dir=args.quant_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        lora_dir=args.lora_dir,
        lora_scales=args.lora_scale,
        quant=args.quant,
        dit_path=args.dit_path,
        infinitetalk_dir=args.infinitetalk_dir,
    )

    if args.num_persistent_param_in_dit is not None:
        wan_i2v.vram_management = True
        wan_i2v.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit)

    # Initialize models once before the loop to prevent race conditions
    wav2vec_feature_extractor, audio_encoder = custom_init("cpu", args.wav2vec_dir)

    job_file = os.path.join(args.video_dir, "job.txt")
    while True:
        try:
            time.sleep(10.0)
            if not os.path.exists(job_file):
                time.sleep(1.0)
                continue

            job_to_process = None
            if rank == 0:
                with open(job_file, "r+", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        lines = [line.strip() for line in f if line.strip()]
                        updated_lines = []
                        job_found = False
                        for line in lines:
                            parts = line.strip().split(args.sep)
                            if not job_found and len(parts) >= 17 and parts[1] == "queued":
                                job_found = True
                                parts[1] = "processing"  # Mark as processing
                                parts[15] = str(int(time.time()))  # Set start time
                                job_to_process = parts
                                updated_lines.append(args.sep.join(map(str, parts)) + "\n")
                            else:
                                updated_lines.append(line + "\n")

                        if job_found:
                            f.seek(0)
                            f.truncate()
                            f.writelines(updated_lines)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)

            if world_size > 1:
                job_list = [job_to_process] if rank == 0 else [None]
                dist.broadcast_object_list(job_list, src=0)
                job_to_process = job_list[0]

            if job_to_process:
                try:
                    id, status, created_str, prompt, seconds, size, quality, fps, shift, steps, guide_scale, audio_guide_scale, seed, logo_video, generate_duration, start_time, end_time, *error_msg_parts = job_to_process
                    generate_start_time = float(start_time)

                    fps = 25
                    user_frames = int(seconds) * fps + 1
                    num_frames = 81 if user_frames >= 81 else find_max_matching_frame(user_frames, 5)
                    generated_list = []
                    job_dir = os.path.join(args.video_dir, id)
                    os.makedirs(job_dir, exist_ok=True)
                    input_json = os.path.join(job_dir, "input.json")
                    audio_save_dir = os.path.join(job_dir, "audio")
                    save_file = os.path.join(job_dir, "output")
                    with open(input_json, "r", encoding="utf-8") as f:
                        input_data = json.load(f)

                        audio_save_dir = os.path.join(audio_save_dir, input_data["cond_video"].split("/")[-1].split(".")[0])
                        os.makedirs(audio_save_dir, exist_ok=True)

                        conds_list = []

                        if args.scene_seg and is_video(input_data["cond_video"]):
                            time_list, cond_list = shot_detect(input_data["cond_video"], audio_save_dir)
                            if len(time_list) == 0:
                                conds_list.append([input_data["cond_video"]])
                                conds_list.append([input_data["cond_audio"]["person1"]])
                                if len(input_data["cond_audio"]) == 2:
                                    conds_list.append([input_data["cond_audio"]["person2"]])
                            else:
                                audio1_list = split_wav_librosa(input_data["cond_audio"]["person1"], time_list, audio_save_dir)
                                conds_list.append(cond_list)
                                conds_list.append(audio1_list)
                                if len(input_data["cond_audio"]) == 2:
                                    audio2_list = split_wav_librosa(input_data["cond_audio"]["person2"], time_list, audio_save_dir)
                                    conds_list.append(audio2_list)
                        else:
                            conds_list.append([input_data["cond_video"]])
                            conds_list.append([input_data["cond_audio"]["person1"]])
                            if len(input_data["cond_audio"]) == 2:
                                conds_list.append([input_data["cond_audio"]["person2"]])

                        if len(input_data["cond_audio"]) == 2:
                            new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(input_data["cond_audio"]["person1"], input_data["cond_audio"]["person2"], input_data["audio_type"])
                            sum_audio = os.path.join(audio_save_dir, "sum_all.wav")
                            sf.write(sum_audio, sum_human_speechs, 16000)
                            input_data["video_audio"] = sum_audio
                        else:
                            human_speech = audio_prepare_single(input_data["cond_audio"]["person1"])
                            sum_audio = os.path.join(audio_save_dir, "sum_all.wav")
                            sf.write(sum_audio, human_speech, 16000)
                            input_data["video_audio"] = sum_audio
                        logging.info("Generating video ...")

                        for idx, items in enumerate(zip(*conds_list)):
                            input_clip = {}
                            input_clip["prompt"] = input_data.get("prompt", " ")
                            input_clip["cond_video"] = items[0]

                            if "audio_type" in input_data:
                                input_clip["audio_type"] = input_data["audio_type"]
                            if "bbox" in input_data:
                                input_clip["bbox"] = input_data["bbox"]
                            cond_audio = {}
                            if args.audio_mode == "localfile":
                                if len(input_data["cond_audio"]) == 2:
                                    new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(items[1], items[2], input_data["audio_type"])
                                    audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                                    audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                                    sum_audio = os.path.join(audio_save_dir, "sum.wav")
                                    sf.write(sum_audio, sum_human_speechs, 16000)
                                    cond_audio["person1"] = audio_embedding_1
                                    cond_audio["person2"] = audio_embedding_2
                                    input_clip["video_audio"] = sum_audio
                                elif len(input_data["cond_audio"]) == 1:
                                    human_speech = audio_prepare_single(items[1])
                                    audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
                                    sum_audio = os.path.join(audio_save_dir, "sum.wav")
                                    sf.write(sum_audio, human_speech, 16000)
                                    cond_audio["person1"] = audio_embedding
                                    input_clip["video_audio"] = sum_audio

                            input_clip["cond_audio"] = cond_audio

                            video = wan_i2v.generate_infinitetalk(
                                input_clip,
                                size_buckget=args.size,
                                motion_frame=args.motion_frame,
                                frame_num=num_frames,
                                shift=float(shift),
                                sampling_steps=int(steps),
                                text_guide_scale=float(guide_scale),
                                audio_guide_scale=float(audio_guide_scale),
                                seed=int(seed),
                                offload_model=args.offload_model,
                                max_frames_num=args.max_frame_num,
                                color_correction_strength=args.color_correction_strength,
                                extra_args=args,
                            )

                            generated_list.append(video)

                        if rank == 0:
                            sum_video = torch.cat(generated_list, dim=1)
                            if logo_video.lower() == "true":
                                save_video_with_logo(sum_video, save_file, [input_data["video_audio"]], high_quality_save=False, fps=fps)
                            else:
                                save_video_ffmpeg(sum_video, save_file, [input_data["video_audio"]], high_quality_save=False, fps=fps)

                    generate_end_time = time.time()
                    job_processed = [id, "completed", created_str, prompt, seconds, size, quality, fps, shift, steps, guide_scale, audio_guide_scale, seed, logo_video, max(0, int(generate_end_time - generate_start_time)), int(generate_start_time), int(generate_end_time), ""]
                    if rank == 0:
                        update_job(job_processed, args)
                except Exception as e:
                    logging.error(f"error: {e}")
                    generate_end_time = time.time()
                    job_processed = [id, "error", created_str, prompt, seconds, size, quality, fps, shift, steps, guide_scale, audio_guide_scale, seed, logo_video, max(0, int(generate_end_time - generate_start_time)), int(generate_start_time), int(generate_end_time), str(e)]
                    if rank == 0:
                        update_job(job_processed, args)

        except Exception as e:
            logging.error(f"Job worker encountered an error: {e}")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
