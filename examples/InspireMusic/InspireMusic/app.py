# Copyright (c) 2024 Alibaba Inc (authors: Chong Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.system('nvidia-smi')
os.system('apt update -y && apt-get install -y apt-utils && apt install -y unzip')
os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'
os.system('mkdir pretrained_models && cd pretrained_models && git clone https://huggingface.co/FunAudioLLM/InspireMusic-Base.git &&git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git &&git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B.git &&git clone https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz.git &&git clone https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz.git && for i in InspireMusic-Base InspireMusic-Base-24kHz InspireMusic-1.5B InspireMusic-1.5B-24kHz InspireMusic-1.5B-Long; do sed -i -e "s/\.\.\/\.\.\///g" ${i}/inspiremusic.yaml; done && cd ..')

import sys
import torch
print(torch.backends.cudnn.version())

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import spaces
import gradio as gr
from inspiremusic.cli.inference import InspireMusicModel, env_variables
import torchaudio
import datetime
import hashlib
import importlib

MODELS = ["InspireMusic-1.5B-Long", "InspireMusic-1.5B", "InspireMusic-Base", "InspireMusic-1.5B-24kHz", "InspireMusic-Base-24kHz"]
AUDIO_PROMPT_DIR = "demo/audio_prompts"
OUTPUT_AUDIO_DIR = "demo/outputs"

DEMO_TEXT_PROMPTS = ["Jazz music with drum beats.",
					 "A captivating classical piano performance, this piece exudes a dynamic and intense atmosphere, showcasing intricate and expressive instrumental artistry.",
					 "A soothing instrumental piece blending elements of light music and pop, featuring a gentle guitar rendition. The overall feel is serene and reflective, likely instrumental with no vocals.",
					 "The instrumental rock piece features dynamic oscillations and wave-like progressions, creating an immersive and energetic atmosphere. The music is purely instrumental, with no vocals, and it blends elements of rock and post-rock for a powerful and evocative experience.",
					 "The classical instrumental piece exudes a haunting and evocative atmosphere, characterized by its intricate guitar work and profound emotional depth.",
					 "Experience a dynamic blend of instrumental electronic music with futuristic house vibes, featuring energetic beats and a captivating rhythm. The tracks are likely instrumental, focusing on the immersive soundscapes rather than vocal performances."]

def generate_filename():
	hash_object = hashlib.sha256(str(int(datetime.datetime.now().timestamp())).encode())
	hash_string = hash_object.hexdigest()
	return hash_string

def get_args(
		task, text="", audio=None, model_name="InspireMusic-Base",
		chorus="intro",
		output_sample_rate=48000, max_generate_audio_seconds=30.0, time_start = 0.0, time_end=30.0, trim=False):

	if "24kHz" in model_name:
		output_sample_rate = 24000

	if output_sample_rate == 24000:
		fast = True
	else:
		fast = False
	# This function constructs the arguments required for InspireMusic
	args = {
		"task"                      : task,
		"text"                      : text,
		"audio_prompt"              : audio,
		"model_name"                : model_name,
		"chorus"                    : chorus,
		"fast"                      : fast,
		"fade_out"                  : True,
		"trim"                      : trim,
		"output_sample_rate"        : output_sample_rate,
		"min_generate_audio_seconds": 10.0,
		"max_generate_audio_seconds": max_generate_audio_seconds,
		"max_audio_prompt_length": 5.0,
		"model_dir"                 : os.path.join("pretrained_models",
												   model_name),
		"result_dir"                : OUTPUT_AUDIO_DIR,
		"output_fn"                 : generate_filename(),
		"format"                    : "wav",
		"time_start" : time_start,
		"time_end": time_end,
		"fade_out_duration": 1.0,
	}

	if args["time_start"] is None:
		args["time_start"] = 0.0
	args["time_end"] = args["time_start"] + args["max_generate_audio_seconds"]

	print(args)
	return args


def trim_audio(audio_file, cut_seconds=5):
	audio, sr = torchaudio.load(audio_file)
	num_samples = cut_seconds * sr
	cutted_audio = audio[:, :num_samples]
	output_path = os.path.join(AUDIO_PROMPT_DIR, "audio_prompt_" + generate_filename() + ".wav")
	torchaudio.save(output_path, cutted_audio, sr)
	return output_path

@spaces.GPU(duration=120)
def music_generation(args):
	env_variables()
	model = InspireMusicModel(
			model_name=args["model_name"],
			model_dir=args["model_dir"],
			min_generate_audio_seconds=args["min_generate_audio_seconds"],
			max_generate_audio_seconds=args["max_generate_audio_seconds"],
			sample_rate=24000,
			output_sample_rate=args["output_sample_rate"],
			load_jit=True,
			load_onnx=False,
			fast=args["fast"],
			result_dir=args["result_dir"])

	output_path = model.inference(
			task=args["task"],
			text=args["text"],
			audio_prompt=args["audio_prompt"],
			chorus=args["chorus"],
			time_start=args["time_start"],
			time_end=args["time_end"],
			output_fn=args["output_fn"],
			max_audio_prompt_length=args["max_audio_prompt_length"],
			fade_out_duration=args["fade_out_duration"],
			output_format=args["format"],
			fade_out_mode=args["fade_out"],
			trim=args["trim"])
	return output_path


def demo_inspiremusic_t2m(text, model_name, chorus,
					 output_sample_rate, max_generate_audio_seconds):
	args = get_args(
			task='text-to-music', text=text, audio=None,
			model_name=model_name, chorus=chorus,
			output_sample_rate=output_sample_rate,
			max_generate_audio_seconds=max_generate_audio_seconds)
	return music_generation(args)

def demo_inspiremusic_con(text, audio, model_name, chorus,
					 output_sample_rate, max_generate_audio_seconds):
	args = get_args(
			task='continuation', text=text, audio=trim_audio(audio, cut_seconds=5),
			model_name=model_name, chorus=chorus,
			output_sample_rate=output_sample_rate,
			max_generate_audio_seconds=max_generate_audio_seconds)
	return music_generation(args)

def main():
	with gr.Blocks(theme=gr.themes.Soft()) as demo:
		gr.Markdown("""
		# InspireMusic
		- Support music generation tasks with long-form and high audio quality, sampling rates up to 48kHz. 
		- Github: https://github.com/FunAudioLLM/InspireMusic/  |  ModelScope Studio: https://modelscope.cn/studios/iic/InspireMusic
		- Available music generation models: [InspireMusic-1.5B-Long](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long), [InspireMusic-1.5B](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B), [InspireMusic-Base](https://huggingface.co/FunAudioLLM/InspireMusic-Base), [InspireMusic-1.5B-24kHz](https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-24kHz), [InspireMusic-Base-24kHz](https://huggingface.co/FunAudioLLM/InspireMusic-Base-24kHz). Both on Huggingface and ModelScope.
		- Currently only support English text prompts.
		- This page is for demo purpose, if you want to generate long-form audio, e.g., 5mins, please try to deploy locally. Thank you for your support.
		""")

		with gr.Row(equal_height=True):
			model_name = gr.Dropdown(
					MODELS, label="Select Model Name",
					value="InspireMusic-1.5B-Long")
			chorus = gr.Dropdown(["intro", "verse", "chorus", "outro"],
								 label="Chorus Mode", value="intro")
			output_sample_rate = gr.Dropdown([48000, 24000],
											 label="Output Audio Sample Rate (Hz)",
											 value=48000)
			max_generate_audio_seconds = gr.Slider(10, 300,
												   label="Generate Audio Length (s)",
												   value=30)

		with gr.Row(equal_height=True):
			text_input = gr.Textbox(label="Input Text (For Text-to-Music Task)",
									value="Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance.")

			audio_input = gr.Audio(
				label="Input Audio Prompt (For Music Continuation Task)",
				type="filepath")
		music_output = gr.Audio(label="Generated Music", type="filepath", autoplay=True, show_download_button = True)

		with gr.Row():
			button = gr.Button("Start Text-to-Music Task")
			button.click(demo_inspiremusic_t2m,
						 inputs=[text_input, model_name,
								 chorus,
								 output_sample_rate,
								 max_generate_audio_seconds],
						 outputs=music_output)

			generate_button = gr.Button("Start Music Continuation Task")
			generate_button.click(demo_inspiremusic_con,
								  inputs=[text_input, audio_input, model_name,
										  chorus,
										  output_sample_rate,
										  max_generate_audio_seconds],
								  outputs=music_output)
		t2m_examples = gr.Examples(examples=DEMO_TEXT_PROMPTS, inputs=[text_input])
	demo.launch()

if __name__ == '__main__':
	os.makedirs(AUDIO_PROMPT_DIR, exist_ok=True)
	os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
	main()
