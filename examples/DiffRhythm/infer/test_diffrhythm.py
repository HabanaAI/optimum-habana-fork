# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import time as tm_perf
import random
import soundfile as sf
import numpy as np

import torch
from einops import rearrange

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.gpu_migration

print("Current working directory:", os.getcwd())

def set_seed():
    seed = 5451
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


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

time_box = time_box_t()

def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    song_duration,
    chunked=False,
):
    time_box.start()
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            max_duration=duration,
            song_duration=song_duration, 
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num
        )
        time_box.show_time('cfm sample')

        outputs = []
        outputs_tmp = []
        for latent in latents:
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)  # [b d t]

            output = decode_audio(latent, vae_model, chunked=chunked)

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            # Peak normalize, clip, convert to int16, and save to file
            #output = (
            #    output.to(torch.float32)
            #    .div(torch.max(torch.abs(output)))
            #    .clamp(-1, 1)
            #    .mul(32767)
            #    .to(torch.int16)
            #    .cpu()
            #)
            #outputs.append(output)
            outputs_tmp.append(output)

        for output in outputs_tmp:
            output = (
                output.to(torch.float32)
                .div(torch.max(torch.abs(output)))
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )
            outputs.append(output)

        time_box.show_time('vae decode')

        return outputs

def test_main(args, cfm, tokenizer, muq, vae, max_frames):
    if args.lrc_path:
        with open(args.lrc_path, "r", encoding='utf-8') as f:
            lrc = f.read()
    else:
        lrc = ""
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(max_frames, lrc, tokenizer, args.audio_length, device)

    if args.ref_audio_path:
        style_prompt = get_style_prompt(muq, args.ref_audio_path)
    else:
        style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)

    negative_style_prompt = get_negative_style_prompt(device)

    latent_prompt, pred_frames = get_reference_latent(device, max_frames, args.edit, args.edit_segments, args.ref_song, vae)

    for i in range(args.loop):
        s_t = time.time()
        generated_songs = inference(
            cfm_model=cfm,
            vae_model=vae,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=end_frame,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            pred_frames=pred_frames,
            chunked=args.chunked,
            batch_infer_num=args.batch_infer_num,
            song_duration=song_duration
        )
        torch.hpu.synchronize()
        duration = time.time() - s_t
        print("DiffRhythm Pipeline Latency in Loop #{:d}: {:.1f} sec".format(i, duration))
    
    generated_song = random.sample(generated_songs, 1)[0]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{device}_len_{args.audio_length}.wav")
    sf.write(output_path, generated_song.t(), 44100, subtype='PCM_16')
    print(f'save {output_path} done')

if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lrc-path",
        type=str,
        help="lyrics of target song",
    )  # lyrics of target song
    parser.add_argument(
        "--ref-prompt",
        type=str,
        help="reference prompt as style prompt for target song",
        required=False,
    )  # reference prompt as style prompt for target song
    parser.add_argument(
        "--ref-audio-path",
        type=str,
        help="reference audio as style prompt for target song",
        required=False,
    )  # reference audio as style prompt for target song
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="whether to use chunked decoding",
    )  # whether to use chunked decoding
    parser.add_argument(
        "--audio-length",
        type=int,
        default=95,
        # choices=[95, 285],
        help="length of generated song, upported values are exactly 95 or any value between 96 and 285 (inclusive).",
    )  # length of target song
    # parser.add_argument(
    #     "--repo-id", type=str, default="ASLP-lab/DiffRhythm-base", help="target model"
    # )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="infer/example/output",
        help="output directory fo generated song",
    )  # output directory of target song
    parser.add_argument(
        "--edit",
        action="store_true",
        help="whether to open edit mode",
    )  # edit flag
    parser.add_argument(
        "--ref-song",
        type=str,
        required=False,
        help="reference prompt as latent prompt for editing",
    )  # reference prompt as latent prompt for editing
    parser.add_argument(
        "--edit-segments",
        type=str,
        required=False,
        help="Time segments to edit (in seconds). Format: `[[start1,end1],...]`. "
             "Use `-1` for audio start/end (e.g., `[[-1,25], [50.0,-1]]`)."
    )  # edit segments of target song
    parser.add_argument(
        "--batch-infer-num",
        type=int,
        default=1,
        required=False,
        help="number of songs per batch",
    )  # number of songs per batch
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Number of benchmark loops for generation.",
    )
    args = parser.parse_args()

    assert (
        args.ref_prompt or args.ref_audio_path
    ), "either ref_prompt or ref_audio_path should be provided"
    assert not (
        args.ref_prompt and args.ref_audio_path
    ), "only one of them should be provided"
    if args.edit:
        assert (
            args.ref_song and args.edit_segments
        ), "reference song and edit segments should be provided for editing"

    device = "cpu"
    if torch.cuda.is_available():
        device = "hpu"
    elif torch.mps.is_available():
        device = "mps"

    audio_length = args.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif 95 < audio_length <= 285:
        max_frames = 6144
    else:
        raise ValueError(
            f"Invalid audio_length: {audio_length}. "
            "Supported values are exactly 95 or any value between 96 and 285 (inclusive)."
        )

    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    vae.forward = vae.decode_export
    #vae = ht.hpu.wrap_in_hpu_graph(vae)

    for audio_length in range(120, 135):
        args.audio_length = audio_length
        test_main(args, cfm, tokenizer, muq, vae, max_frames)

