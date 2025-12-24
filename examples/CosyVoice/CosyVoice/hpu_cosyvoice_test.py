
import os
# Enable lazy mode for HPU
os.environ['PT_HPU_LAZY_MODE'] = '1'

import sys
import torch
import torchaudio
import habana_frameworks.torch as ht_torch
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import time

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

logging.getLogger('numba').setLevel(logging.WARNING)

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

cv_examples = [
    [
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        '极速复刻',
        None,
        "./asset/zero_shot_prompt.wav",
        "希望你以后能够做的比我还好呦。",
        "",
    ],
    [
        "And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.",
        '跨语种复刻',
        None,
        "./asset/cross_lingual_prompt.wav",
        "",
        "",
    ],
    [
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        '自然语言控制',
        None,
        "./asset/zero_shot_prompt.wav",
        "",
        "用四川话说这句话",
    ],
    [
        "这次机会让我能够在新的领域中不断学习和成长，同时也激励我去克服自身的不足。",
        '预训练音色',
        None,  # sft_spk[0], use None here, can be replaced with actual speaker id
        None,
        "",
        "",
    ],
]


#print(cosyvoice.model.flow.decoder.estimator.down_blocks)
#exit()

torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """
    Post-process audio: trim silence and normalize amplitude.
    """
    import librosa
    max_val = 0.8
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def warmup_audio(tts_text, prompt_text, prompt_wav_path):
    """
    Warmup process before formal tests, using zero-shot inference.
    """
    prompt_speech_16k = postprocess(load_wav(prompt_wav_path, 16000))
    from cosyvoice.utils.common import set_all_random_seed
    set_all_random_seed(0)
    for _ in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
        continue

def run_cv_examples():
    """
    Batch test cv_examples from webui.py.
    """
    sft_spk_list = cosyvoice.list_available_spks()
    output_dir = 'output/cosyvoice.inference/'
    os.makedirs(output_dir, exist_ok=True)
    for idx, example in enumerate(cv_examples):
        tts_text, mode, sft_spk, prompt_wav_path, prompt_text, instruct_text = example
        print(f"\n===== Running Example {idx+1}: mode={mode} =====")
        prompt_speech_16k = None
        if prompt_wav_path:
            prompt_speech_16k = load_wav(prompt_wav_path, 16000)
        # 统一输出文件名
        mode_map = {
            '预训练音色': 'pretrain',
            '极速复刻': 'zero_shot',
            '跨语种复刻': 'cross_lingual',
            '自然语言控制': 'instruct',
        }
        out_path = os.path.join(output_dir, f'{mode_map.get(mode, mode)}_{idx}.wav')
        audio_cat = None
        # Pretrained speaker mode
        if mode == '预训练音色':
            zero_shot_spk_id = sft_spk if sft_spk else (sft_spk_list[0] if sft_spk_list else '')
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=zero_shot_spk_id, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
        # Zero-shot mode
        elif mode == '极速复刻':
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
        # Cross-lingual mode
        elif mode == '跨语种复刻':
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
        # Instruct mode
        elif mode == '自然语言控制':
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
        if audio_cat is not None:
            torchaudio.save(out_path, audio_cat, cosyvoice.sample_rate)
            print(f"Example {idx+1} finished. Output saved to {out_path}")
        else:
            print(f"Example {idx+1} finished. No audio generated.")

nwarmup = 1
loop = 3
device='hpu'
model = cosyvoice.model.llm.llm.model.bfloat16().eval().to(device)
cosyvoice.model.llm.llm.model = wrap_in_hpu_graph(model)

model = cosyvoice.model.llm.llm_decoder.bfloat16().eval().to(device)
cosyvoice.model.llm.llm_decoder = wrap_in_hpu_graph(model)

cosyvoice.model.flow = cosyvoice.model.flow.bfloat16().eval()


if __name__ == '__main__':
    warmup_audio(cv_examples[0][0], cv_examples[0][4], cv_examples[0][3])
    run_cv_examples()

