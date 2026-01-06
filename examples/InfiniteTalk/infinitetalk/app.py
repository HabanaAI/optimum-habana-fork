# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import sys
import json
import warnings
from datetime import datetime

import gradio as gr
warnings.filterwarnings('ignore')


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument('--server-port', type=int, default=8481, help='Demo server port.')
    parser.add_argument('--cp', type=int, default=8, help='cp size.')
    args = parser.parse_args()
    return args

def run_graio_demo(args):
    # Create pipe
    for i in range(args.cp):
        pipename = f"infinitetalk_fifo_{i}"
        if os.path.exists(pipename):
            os.unlink(pipename)
        os.mkfifo(pipename)

    pipename = "infinitetalk_fifo_result"
    if not os.path.exists(pipename):
        os.mkfifo(pipename)
    # Open pipe
    pipelist = []
    for i in range(args.cp):
        pipename = f"infinitetalk_fifo_{i}"
        pipelist.append(os.open(pipename, os.O_WRONLY))
    result_pipe = os.open("infinitetalk_fifo_result", os.O_RDONLY)

    def broadcast_args(message):
        for pipe in pipelist:
            os.write(pipe, message.encode())

    def read_result():
        return os.read(result_pipe, 16384).decode()

    def generate_video(img2vid_image, vid2vid_vid, task_mode, img2vid_prompt,
                       n_prompt, img2vid_audio_1, img2vid_audio_2,
                       sd_steps, seed, text_guide_scale, audio_guide_scale,
                       mode_selector, max_video_length):
        input_data = {}
        input_data["prompt"] = img2vid_prompt
        if task_mode=='VideoDubbing':
            input_data["cond_video"] = vid2vid_vid
        else:
            if img2vid_image is None:
                gr.Warning("image cannnot be empty")
                return gr.update()
            input_data["cond_video"] = img2vid_image
        person = {}
        if mode_selector == "Single Person(Local File)":
            if img2vid_audio_1 is None:
                gr.Warning("audio cannnot be empty")
                return gr.update()
            person['person1'] = img2vid_audio_1
        elif mode_selector == "Multi Person(Local File, audio add)":
            if img2vid_audio_1 is None or img2vid_audio_2 is None:
                gr.Warning("audio cannnot be empty")
                return gr.update()
            person['person1'] = img2vid_audio_1
            person['person2'] = img2vid_audio_2
            input_data["audio_type"] = 'add'
        elif mode_selector == "Multi Person(Local File, audio parallel)":
            if img2vid_audio_1 is None or img2vid_audio_2 is None:
                gr.Warning("audio cannnot be empty")
                return gr.update()
            person['person1'] = img2vid_audio_1
            person['person2'] = img2vid_audio_2
            input_data["audio_type"] = 'para'

        input_data["cond_audio"] = person

        args = {}
        args['input_data'] = input_data
        args['n_prompt'] = n_prompt
        args['sampling_steps'] = sd_steps
        args['seed'] = seed
        args['text_guide_scale'] = text_guide_scale
        args['audio_guide_scale'] = audio_guide_scale
        args['max_frames_num'] = max_video_length * 25

        print(args)
        broadcast_args(json.dumps(args))
        result_video = read_result()
        print(result_video)

        return result_video

    def toggle_audio_mode(mode):
        if 'Single' in mode:
            return [
                gr.Audio(visible=True, interactive=True),
                gr.Audio(visible=False, interactive=False),
                gr.Textbox(visible=False, interactive=False)
            ]
        else:
            return [
                gr.Audio(visible=True, interactive=True),
                gr.Audio(visible=True, interactive=True),
                gr.Textbox(visible=False, interactive=False)
            ]

    def show_upload(mode):
        if mode == "SingleImageDriven":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)


    with gr.Blocks() as demo:

        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        MeiGen-InfiniteTalk
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        InfiniteTalk: Audio-driven Video Generation for Spare-Frame Video Dubbing.
                    </div>
                    <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                        <a href=''><img src='https://img.shields.io/badge/Project-Page-blue'></a>
                        <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
                        <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
                    </div>


                    """)

        with gr.Row():
            with gr.Column(scale=1):
                task_mode = gr.Radio(
                    choices=["SingleImageDriven"], #, "VideoDubbing"],
                    label="Choose SingleImageDriven task or VideoDubbing task",
                    value="SingleImageDriven"
                )
                vid2vid_vid = gr.Video(
                    label="Upload Input Video",
                    visible=False)
                img2vid_image = gr.Image(
                    type="filepath",
                    label="Upload Input Image",
                    elem_id="image_upload",
                    visible=True,
                    value='examples/single/ref_image.png',
                )
                img2vid_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate",
                )
                task_mode.change(
                    fn=show_upload,
                    inputs=task_mode,
                    outputs=[img2vid_image, vid2vid_vid]
                )

                with gr.Accordion("Audio Options", open=True):
                    mode_selector = gr.Radio(
                        choices=["Single Person(Local File)"],# "Multi Person(Local File, audio add)", "Multi Person(Local File, audio parallel)"],
                        label="Select person and audio mode.",
                        value="Single Person(Local File)"
                    )
                    img2vid_audio_1 = gr.Audio(label="Conditioning Audio for speaker 1", type="filepath", visible=True, value="examples/single/1.wav")
                    img2vid_audio_2 = gr.Audio(label="Conditioning Audio for speaker 2", type="filepath", visible=False)
                    mode_selector.change(
                        fn=toggle_audio_mode,
                        inputs=mode_selector,
                        outputs=[img2vid_audio_1, img2vid_audio_2]
                    )

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        sd_steps = gr.Slider(
                            label="Diffusion steps",
                            minimum=1,
                            maximum=1000,
                            value=20,
                            step=1)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=42)
                    with gr.Row():
                        text_guide_scale = gr.Slider(
                            label="Text Guide scale",
                            minimum=0,
                            maximum=20,
                            value=1.0,
                            step=5)
                        audio_guide_scale = gr.Slider(
                            label="Audio Guide scale",
                            minimum=0,
                            maximum=20,
                            value=2.0,
                            step=4)
                    with gr.Row():
                        max_video_length = gr.Slider(
                            label="Video Length (seconds)",
                            minimum=1,
                            maximum=1000,
                            value=300,
                            step=1)
                    # with gr.Row():
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe the negative prompt you want to add",
                        value="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
                    )

                run_i2v_button = gr.Button("Generate Video")

            with gr.Column(scale=2):
                result_gallery = gr.Video(
                    label='Generated Video', interactive=False, height=600, )

                gr.Examples(
                    examples = [
                        ['SingleImageDriven', 'examples/single/ref_image.png', None, "A woman is passionately singing into a professional microphone in a recording studio. She wears large black headphones and a dark cardigan over a gray top. Her long, wavy brown hair frames her face as she looks slightly upwards, her mouth open mid-song. The studio is equipped with various audio equipment, including a mixing console and a keyboard, with soundproofing panels on the walls. The lighting is warm and focused on her, creating a professional and intimate atmosphere. A close-up shot captures her expressive performance.", "Single Person(Local File)", "examples/single/1.wav", None],
                    ],
                    inputs = [task_mode, img2vid_image, vid2vid_vid, img2vid_prompt, mode_selector, img2vid_audio_1, img2vid_audio_2],
                )


        run_i2v_button.click(
            fn=generate_video,
            inputs=[img2vid_image, vid2vid_vid, task_mode,
                    img2vid_prompt, n_prompt, img2vid_audio_1,
                    img2vid_audio_2,sd_steps, seed, text_guide_scale,
                    audio_guide_scale, mode_selector, max_video_length],
            outputs=[result_gallery],
        )
    demo.queue(max_size=8, default_concurrency_limit=1).launch( server_name="0.0.0.0", server_port=args.server_port)


if __name__ == "__main__":
    args = _parse_args()
    run_graio_demo(args)
