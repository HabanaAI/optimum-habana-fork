from hmac import new
import sys
import os
import argparse

import time
import json
import torch
import torchaudio
from habana_frameworks import torch as ht
import habana_frameworks.torch.gpu_migration

import numpy as np
from omegaconf import OmegaConf
from codeclm.models import builders
import gc
from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from third_party.demucs.models.pretrained import get_model_from_yaml
import re

auto_prompt_type = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']

class Separator:
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model

    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if (fs != 48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        if a.shape[-1] >= 48000*10:
            a = a[..., :48000*10]
        return a[:, 0:48000*10]

    def run(self, audio_path, output_dir='tmp', ext=".flac"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:  # 4
            vocal_path = output_paths[0]
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        full_len = full_audio.shape[1]
        vocal_len = vocal_audio.shape[1]
        if full_len > vocal_len:
            full_audio = full_audio[:,:vocal_len]
        else:
            vocal_audio = vocal_audio[:,:full_len]
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio

class API_Model:
    def __init__(self, ckpt_path, use_flash_attn):
        torch.set_num_threads(1)
        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        ckpt_path = os.path.join(ckpt_path, 'model.pt')
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        cfg.lm.use_flash_attn_2 = use_flash_attn
        max_duration = cfg.max_dur
        cfg.mode = 'inference'

        self.separator = Separator()
        audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
        self.audio_tokenizer = audio_tokenizer.eval()
        self.auto_prompt = torch.load('tools/new_prompt.pt')

        if "audio_tokenizer_checkpoint_sep" in cfg.keys():
            seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
        else:
            seperate_tokenizer = None

        if seperate_tokenizer is not None:
            self.seperate_tokenizer = seperate_tokenizer.eval().cuda()

        audiolm = builders.get_lm_model(cfg)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval()
        self.audiolm = audiolm.cuda().to(torch.bfloat16)

        self.model = CodecLM(name = "tmp",
            lm = audiolm,
            audiotokenizer = None,
            max_duration = max_duration,
            seperate_tokenizer = seperate_tokenizer,
        )
        self.max_duration = max_duration
        self.sample_rate = cfg.sample_rate

    def generate(self, item, output_path, bgm_path, vocal_path):
        target_wav_name = output_path
        gen_type = item['gen_type']
        # get prompt audio
        if "prompt_audio_path" in item:
            assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
            assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            with torch.no_grad():
                pmt_wav, vocal_wav, bgm_wav = self.separator.run(item['prompt_audio_path'])
            item['raw_pmt_wav'] = pmt_wav
            item['raw_vocal_wav'] = vocal_wav
            item['raw_bgm_wav'] = bgm_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if pmt_wav.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            pmt_wav = list(pmt_wav)
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            if vocal_wav.dim() != 3:
                raise ValueError("Vocal wavs should have a shape [B, C, T].")
            vocal_wav = list(vocal_wav)
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            if bgm_wav.dim() != 3:
                raise ValueError("BGM wavs should have a shape [B, C, T].")
            bgm_wav = list(bgm_wav)
            if type(pmt_wav) == list:
                pmt_wav = torch.stack(pmt_wav, dim=0)
            if type(vocal_wav) == list:
                vocal_wav = torch.stack(vocal_wav, dim=0)
            if type(bgm_wav) == list:
                bgm_wav = torch.stack(bgm_wav, dim=0)
            pmt_wav = pmt_wav
            vocal_wav = vocal_wav
            bgm_wav = bgm_wav
            with torch.no_grad():
                pmt_wav, _ = self.audio_tokenizer.encode(pmt_wav)
            melody_is_wav = False
        elif "auto_prompt_audio_type" in item:
            assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
            prompt_token = self.auto_prompt[item["auto_prompt_audio_type"]][np.random.randint(0, len(self.auto_prompt[item["auto_prompt_audio_type"]]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True
        item['pmt_wav'] = pmt_wav
        item['vocal_wav'] = vocal_wav
        item['bgm_wav'] = bgm_wav
        item['melody_is_wav'] = melody_is_wav
        item["wav_path"] = target_wav_name
        torch.cuda.empty_cache()


        if "prompt_audio_path" in item:
            with torch.no_grad():
                vocal_wav, bgm_wav = self.seperate_tokenizer.encode(item['vocal_wav'].cuda(), item['bgm_wav'].cuda())
            item['vocal_wav'] = vocal_wav
            item['bgm_wav'] = bgm_wav

        torch.cuda.empty_cache()

        cfg_coef = 1.5 #25
        temp = 0.9
        top_k = 50
        top_p = 0.0
        record_tokens = True
        record_window = 50

        self.model.set_generation_params(duration=self.max_duration, extend_stride=5, temperature=temp, cfg_coef=cfg_coef,
                                    top_k=top_k, top_p=top_p, record_tokens=record_tokens, record_window=record_window)

        lyric = item["gt_lyric"]
        descriptions = item["descriptions"] if "descriptions" in item else None
        pmt_wav = item['pmt_wav']
        vocal_wav = item['vocal_wav']
        bgm_wav = item['bgm_wav']
        melody_is_wav = item['melody_is_wav']
        target_wav_name = output_path


        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [descriptions],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }
        start_time = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                tokens = self.model.generate(**generate_inp, return_tokens=True)
        mid_time = time.time()

        with torch.no_grad():
            if 'raw_pmt_wav' in item:
                if gen_type == 'separate':
                    wav_seperate = self.model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='mixed')
                    wav_vocal = self.model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='vocal')
                    wav_bgm = self.model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='bgm')
                elif gen_type == 'mixed':
                    wav_seperate = self.model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'],chunked=True, gen_type=gen_type)
                else:
                    wav_seperate = self.model.generate_audio(tokens,chunked=True, gen_type=gen_type)
                del item['raw_pmt_wav']
                del item['raw_vocal_wav']
                del item['raw_bgm_wav']
            else:
                if gen_type == 'separate':
                    wav_vocal = self.model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = self.model.generate_audio(tokens, chunked=True, gen_type='bgm')
                    wav_seperate = self.model.generate_audio(tokens, chunked=True, gen_type='mixed')
                else:
                    wav_seperate = self.model.generate_audio(tokens, chunked=True, gen_type=gen_type)
        del item['pmt_wav']
        del item['vocal_wav']
        del item['bgm_wav']
        del item['melody_is_wav']
        end_time = time.time()
        if gen_type == 'separate':
            torchaudio.save(vocal_path, wav_vocal[0].cpu().float(), self.sample_rate)
            torchaudio.save(bgm_path, wav_bgm[0].cpu().float(), self.sample_rate)
            torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), self.sample_rate)
        else:
            torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), self.sample_rate)

        print(f"process{target_wav_name}, lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}")

        return

