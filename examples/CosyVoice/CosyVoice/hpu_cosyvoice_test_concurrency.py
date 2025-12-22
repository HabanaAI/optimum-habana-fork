
import os
# Enable lazy mode for HPU
os.environ['PT_HPU_LAZY_MODE'] = '1'
import asyncio
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
from cosyvoice.utils.file_utils import load_wav

cosyvoice = CosyVoice2('/workspace/data/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

cv_examples = [
    [
        "《活着》无疑是一部悲剧性的小说，鲁迅先生说：悲剧是将人生有价值的东西毁灭给人看。"
        "余华用了一个悲剧的故事讲述了生命的价值和活着的意义，同时余华试图用一种委婉和缓的方式来写这个悲剧故事，"
        "余华的目的是希望读者从这个故事中感受到生命的可贵，同时感受到生命的顽强力，希望我们选择坚强而乐观的活着。"
        "福贵的一生经历了巨大的变化，从任性的纨绔子弟到贫穷落魄的孤独老人，他目睹了亲人一个个悲惨离去，"
        "小说在叙述这些故事的时候采用的是一种重复的艺术。",
        '极速复刻',
        None,
        "./asset/ZH_2_prompt.wav",
        "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
        "",
    ],
]


#print(cosyvoice.model.flow.decoder.estimator.down_blocks)
#exit()

torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

async def postprocess(speech, top_db=60, hop_length=220, win_length=440):
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

async def warmup_audio(tts_text, prompt_text, prompt_wav_path):
    """
    Warmup process before formal tests, using zero-shot inference.
    """
    prompt_speech_16k = postprocess(load_wav(prompt_wav_path, 16000))
    from cosyvoice.utils.common import set_all_random_seed
    set_all_random_seed(0)
    for _ in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
        continue

async def run_cv_examples(concurrency, order):
    """
    Batch test cv_examples from webui.py.
    """
    sft_spk_list = cosyvoice.list_available_spks()
    output_dir = f'/workspace/output/cosyvoice-inference/{concurrency}'
    os.makedirs(output_dir, exist_ok=True)
    # 获取全局起点
    global_start = globals().get('_cosyvoice_global_start', None)
    if global_start is None:
        global_start = time.perf_counter()
    total_audio_sec = 0.0
    for idx, example in enumerate(cv_examples):
        tts_text, mode, sft_spk, prompt_wav_path, prompt_text, instruct_text = example
        print(f"\n===== Running Example {idx+1}: mode={mode} , concurrency={concurrency} =====")
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
        out_path = os.path.join(output_dir, f'{mode_map.get(mode, mode)}_{concurrency}_{order}_{idx}.wav')
        audio_cat = None
        sample_audio_sec = 0.0
        # Pretrained speaker mode
        if mode == '预训练音色':
            zero_shot_spk_id = sft_spk if sft_spk else (sft_spk_list[0] if sft_spk_list else '')
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=zero_shot_spk_id, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
                    sample_audio_sec += j['tts_speech'].shape[1] / cosyvoice.sample_rate
        # Zero-shot mode
        elif mode == '极速复刻':
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
                    sample_audio_sec += j['tts_speech'].shape[1] / cosyvoice.sample_rate
        # Cross-lingual mode
        elif mode == '跨语种复刻':
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
                    sample_audio_sec += j['tts_speech'].shape[1] / cosyvoice.sample_rate
        # Instruct mode
        elif mode == '自然语言控制':
            with torch.no_grad():
                for i, j in enumerate(cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=False)):
                    if i == 0:
                        audio_cat = j['tts_speech']
                    else:
                        audio_cat = torch.cat([audio_cat, j['tts_speech']], dim=1)
                    sample_audio_sec += j['tts_speech'].shape[1] / cosyvoice.sample_rate
        # 用全局起点计时
        sample_time = time.perf_counter() - global_start
        total_audio_sec += sample_audio_sec
        # 打印单个样本RTF
        if sample_audio_sec > 0:
            sample_rtf = sample_time / sample_audio_sec
            print(f"[单条RTF-全局起点] idx={idx+1}, time={sample_time:.3f}s, audio={sample_audio_sec:.3f}s, RTF={sample_rtf:.4f}")
            torchaudio.save(out_path, audio_cat, cosyvoice.sample_rate)
            print(f"Example {idx+1} finished. Output saved to {out_path}")
        else:
            print(f"[单条RTF-全局起点] idx={idx+1}, audio=0, 无法计算RTF")
            print(f"Example {idx+1} finished. No audio generated.")
    return total_audio_sec


async def main(result_file_path, CONCURRENCY=1, REQUESTS_PER_CONNECTION=1):
    # CONCURRENCY = 1  # 并发数量
    # REQUESTS_PER_CONNECTION = 1  # 每个连接发送的请求数
    TOTAL_REQUESTS = CONCURRENCY * REQUESTS_PER_CONNECTION
    # 初始化统计信息
    latencies = []
    first_token_times = []
    success_count = 0
    failed_count = 0
    print(f"\n================== Running step {CONCURRENCY} start ====================")
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(current_time)
    tasks = []
    start_time = time.perf_counter()
    # 设置全局起点
    globals()['_cosyvoice_global_start'] = start_time

    # 创建并发任务
    for i in range(TOTAL_REQUESTS):
        task = asyncio.create_task(run_cv_examples(CONCURRENCY, i))
        tasks.append(task)

    # 收集结果
    results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    # 统计所有任务的音频总时长
    total_audio_sec = sum(results)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(current_time)
    print(f"\n===================== Running step {CONCURRENCY} over, total time: {total_time} =======================")

    # 吞吐量计算
    throughput = success_count / total_time

    # 平均每秒请求数计算
    average_requests_per_second = TOTAL_REQUESTS / total_time

    # 总RTF
    if total_audio_sec > 0:
        total_rtf = total_time / total_audio_sec
        print(f"total_time: {total_time:.3f}s, total_audio: {total_audio_sec:.3f}s, RTF: {total_rtf:.4f}")
    else:
        print("total_audio_sec=0, 无法计算RTF")


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


    result_file_path = "report_rerank_2k_1104.xlsx"
    concurrency = [1, 10]
    # concurrency = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    # concurrency = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    # concurrency = [1, 50, 100, 200, 500, 1000]
    # concurrency = [1]
    for case in concurrency:
        requests_per_connection = 1
        asyncio.run(main(result_file_path, case, requests_per_connection))
