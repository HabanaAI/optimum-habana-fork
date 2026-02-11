#!/usr/bin/env python3
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

import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch
import numpy as np
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def job(utt_list, token_list, parquet_file, utt2text, utt2time, utt2chorus, semantic_token_list):
    start_time = time.time()

    text_list = [utt2text[utt] for utt in utt_list]
    time_start = [utt2time[utt][0] for utt in utt_list]
    time_end = [utt2time[utt][1] for utt in utt_list]
    chorus_list = [utt2chorus[utt] for utt in utt_list]
    print(len(token_list))
    print(len(semantic_token_list))
    try:
        df = pd.DataFrame()
        df['utt'] = utt_list
        df['text'] = text_list
        df['chorus'] = chorus_list
        df['time_start'] = time_start
        df['time_end'] = time_end
        df["semantic_token"] = semantic_token_list
        df["acoustic_token"] = token_list
        logging.info(f'Starting to save parquet file: {parquet_file}')
        df.to_parquet(parquet_file)
        logging.info(f'Successfully saved parquet file: {parquet_file}')
    except Exception as e:
        logging(f'Error saving parquet file: {e}')
    logging.info('Processing time {}s'.format(time.time() - start_time))

def text_only_job(utt_list, parquet_file, utt2text, utt2time, utt2chorus):
    start_time = time.time()

    text_list = [utt2text[utt] for utt in utt_list]
    time_start = [utt2time[utt][0] for utt in utt_list]
    time_end = [utt2time[utt][1] for utt in utt_list]
    chorus_list = [utt2chorus[utt] for utt in utt_list]

    try:
        # 保存到parquet
        df = pd.DataFrame()
        df['utt'] = utt_list
        df['text'] = text_list
        df['chorus'] = chorus_list
        df['time_start'] = time_start
        df['time_end'] = time_end
        logging.info(f'Starting to save parquet file: {parquet_file}')
        df.to_parquet(parquet_file)
        logging.info(f'Successfully saved parquet file: {parquet_file}')
    except Exception as e:
        logging(f'Error saving parquet file: {e}')
    logging.info('Processing time {}s'.format(time.time() - start_time))

def parse_trans(line):
    music_structure_labels = ["intro", "verse1", "chorus", "verse2", "verse", "outro"]
    uid,l = line.strip().split("\t")
    split = l.split("|><|")
    time_start = float(split[0].replace("<|",""))
    time_end = float(split[-1].replace("|>", ""))
    chorus = split[1] 
    if split[2] == "lyrics":
        text = "<|lyrics|> " + split[3]
    elif split[2] == "music":
        text = "<|music|>"
    else:
        text = split[2]
    if chorus not in music_structure_labels:
        chorus = random.choice(music_structure_labels)
    if chorus in ["verse1", "verse2"]:
        chorus = "verse"

    if len(split) < 4 or time_start >= time_end:
        print(line, split, time_start, time_end)
        return None
    if time_start < 0:
        time_start = 0.0
    return (uid, time_start, time_end, chorus, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        required=False,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        required=False,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str, required=True)
    parser.add_argument('--des_dir',
                        type=str, required=True)
    parser.add_argument('--semantic_token_dir',
                        type=str,
                        default=None, required=False)
    parser.add_argument('--acoustic_token_dir',
                        type=str,
                        default=None, required=False)
    args = parser.parse_args()

    parquet_list = []
    cnt = 0
    utt2text = {}
    utt2time = {}
    utt2chorus = {}
    uid_list = []

    print(args)

    if not os.path.exists(f'{args.src_dir}/text'):
        raise FileNotFoundError(
            f"Please check: {args.src_dir}/text file does not exist")

    with open(f'{args.src_dir}/text', 'r') as f:
        for l in f:
            res = parse_trans(l)
            if res is None:
                continue
            uid, time_start, time_end, chorus, text = res
            uid_list.append(uid)
            utt2time[uid] = (time_start, time_end)
            utt2chorus[uid] = chorus
            utt2text[uid] = text
    utt2semantic_token = None
    utt2acoustic_token = None
    if args.semantic_token_dir is not None:
        utt2semantic_token = {}
        for fn in os.listdir(args.semantic_token_dir):
            if fn.endswith("pt") and fn.startswith("utt2semantic_"):
                print(f"Starting {fn}")
                try:
                    utt2semantic_token.update(
                        torch.load('{}/{}'.format(args.semantic_token_dir, fn)))
                except:
                    print('{}/{} failed'.format(args.semantic_token_dir, fn))
                    pass
        print(len(utt2semantic_token))

    # # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    if args.acoustic_token_dir is not None:
        for fn in os.listdir(args.acoustic_token_dir):
            if fn.endswith("pt") and fn.startswith("utt2acoustic_"):
                print(f"Starting {fn}")
                utt2token = torch.load(
                    '{}/{}'.format(args.acoustic_token_dir, fn))

                utts = [utt for utt in utt2token.keys() if utt in utt2text.keys()]
                if utt2semantic_token:
                    utts = [utt for utt in utts if
                            utt in utt2semantic_token.keys()]

                if len(utts) == 0:
                    print("0 lines remained.")
                    continue
                if isinstance(utt2token[utts[0]], np.ndarray):
                    token_lists = [utt2token[utt][0].tolist() for utt in utts]
                else:
                    token_lists = [
                        utt2token[utt].tolist() if utt2token[
                                                       utt].dim() == 2 else
                        utt2token[utt][0].tolist()
                        for utt in utts
                    ]
                print(len(token_lists))
                semantic_token_lists = [
                    utt2semantic_token[utt].tolist() if not isinstance(
                            utt2semantic_token[utt], list) else
                    utt2semantic_token[utt] for utt in
                    utts] if utt2semantic_token else None
                for i, j in enumerate(
                        range(0, len(utts), args.num_utts_per_parquet)):
                    parquet_file = os.path.join(args.des_dir,
                                                'parquet_{:09d}.tar'.format(
                                                    cnt + i))
                    print(f"process {parquet_file}")
                    parquet_list.append(parquet_file)
                    token_list = token_lists[j: j + args.num_utts_per_parquet]
                    if semantic_token_lists:
                        semantic_token_list = semantic_token_lists[
                                              j: j + args.num_utts_per_parquet]
                    else:
                        semantic_token_list = None
                    pool.apply_async(job, (
                    utts[j: j + args.num_utts_per_parquet], token_list,
                    parquet_file, utt2text, utt2time, utt2chorus,
                    semantic_token_list))
                    cnt += i

    if args.semantic_token_dir is None and args.acoustic_token_dir is None:
        for i, j in enumerate(
                range(0, len(uid_list), args.num_utts_per_parquet)):
            parquet_file = os.path.join(args.des_dir,
                                        'parquet_{:09d}.tar'.format(cnt + i))
            print(f"process {parquet_file}")
            parquet_list.append(parquet_file)
            pool.apply_async(text_only_job, (
            uid_list[j: j + args.num_utts_per_parquet], parquet_file, utt2text,
            utt2time, utt2chorus))
            cnt += i

    pool.close()
    pool.join()
    print("DONE")

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1:
        for name in parquet_list:
            f1.write(name + '\n')
