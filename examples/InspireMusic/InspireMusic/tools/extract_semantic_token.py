#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc
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
import argparse
import logging
import torch
from tqdm import tqdm
import numpy as np
import torchaudio
import time
import os
from inspiremusic.wavtokenizer.decoder.pretrained import WavTokenizer
from inspiremusic.utils.audio_utils import split_wav_into_chunks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def main(args):
    audio_min_length = 1.0
    audio_max_length = 30.0
    max_chunk_size = int(args.sample_rate * audio_max_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    wavtokenizer = WavTokenizer.from_pretrained_feat(args.config_path, args.ckpt_path).to(device)
    bandwidth_id = torch.tensor([0]).to(device)
    start_time = time.time()    
    utt2semantic_token = {}
    for utt in tqdm(utt2wav.keys()):
        audio, sample_rate = torchaudio.load(utt2wav[utt])

        if sample_rate != args.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=args.sample_rate)
        audio_length = audio.shape[1]
        if audio_length > args.sample_rate * audio_min_length:
            if audio_length > max_chunk_size:
                wav_batch = split_wav_into_chunks(audio_length, audio, max_chunk_size)
                for chunk in wav_batch:
                    chunk = torch.tensor(chunk, dtype=torch.float32).to(device)
                    _, semantic_token = wavtokenizer.encode_infer(chunk, bandwidth_id=bandwidth_id)
                    if semantic_token.is_cuda:
                        semantic_token = semantic_token.cpu()
                    semantic_token = semantic_token.squeeze(0).numpy().astype(np.int16) 
                    if utt not in utt2semantic_token.keys():
                        utt2semantic_token[utt] = semantic_token
                    else:
                        utt2semantic_token[utt] = np.concatenate((utt2semantic_token[utt], semantic_token), axis=1)
            else:
                audio = torch.tensor(audio, dtype=torch.float32).to(device)
                _, semantic_token = wavtokenizer.encode_infer(audio, bandwidth_id=bandwidth_id)
                if semantic_token.is_cuda:
                    semantic_token = semantic_token.cpu()
                semantic_token = semantic_token.squeeze(0).numpy().astype(np.int16) 
                utt2semantic_token[utt] = semantic_token
        else:
            logging.warning('This audio length is too short.')

    torch.save(utt2semantic_token, '{}/utt2semantic_token.pt'.format(args.dir))
    logging.info('spend time {}'.format(time.time() - start_time))


def reconstruct(semantic_token_file, config_path, ckpt_path, outdir, sample_rate=24000):
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bandwidth_id = torch.tensor([0]).to(device)
    wavtokenizer = WavTokenizer.from_pretrained_feat(config_path, ckpt_path).to(device)
    utt2semantic_token = torch.load(semantic_token_file)
    for utt in tqdm(utt2semantic_token.keys()):
        token = utt2semantic_token[utt]
        new_tensor = torch.tensor(token).to(device).unsqueeze(0)
        features = wavtokenizer.codes_to_features(new_tensor)
        wav = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
        wav = wav.cpu().detach()
        torchaudio.save(outdir + "/" + utt + ".wav", wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--config_path',
                        type=str, default="pretrained_models/InspireMusic-Base/wavtokenizer/config.yaml")
    parser.add_argument('--ckpt_path',
                        type=str, default="pretrained_models/InspireMusic-Base/wavtokenizer/model.pt")
    parser.add_argument('--sample_rate',
                        default=24000,
                        type=int)
    parser.add_argument('--outwavdir',
                        type=str, default="./exp/wavs")

    args = parser.parse_args()

    main(args)
