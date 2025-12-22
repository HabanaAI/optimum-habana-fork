# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

# All HPU-related imports and model loading are moved to a child process
# NOTE: Gaudi adaptation is imported and applied inside the HPU child process

inference_mode_list = ['极速复刻', '跨语种复刻', '自然语言控制', '预训练音色']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '极速复刻': '1. 选择参考音频文件，或录入参考音频，注意不超过30s，若同时提供，优先选择参考音频文件\n2. 输入参考文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择参考音频文件，或录入参考音频，注意不超过30s，若同时提供，优先选择参考音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择参考音频文件\n2. 输入控制文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

# ---------------------------
# Subprocess for HPU work
# ---------------------------
import multiprocessing

_task_queue = None
_result_queue = None
_task_event = None
_ready_event = None
_shutdown_event = None
_hpu_ps = None

_sample_rate = 24000  # will be updated from child
_sft_spk_list = ['']  # will be updated from child


def _hpu_child_process(task_queue, result_queue, task_event, shutdown_event, ready_event, args):
    # HPU-specific environment and imports
    os.environ['PT_HPU_LAZY_MODE'] = '1'
    try:
        # Import HPU libraries inside child
        import habana_frameworks.torch as ht_torch  # noqa: F401
        import habana_frameworks.torch.core as htcore  # noqa: F401
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    except Exception as e:
        result_queue.put({"type": "ready", "ok": False, "error": f"HPU import failed: {e}"})
        ready_event.set()
        return

    # Adapt transformers within child
    try:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()
    except Exception as e:
        # Not fatal, but log
        print(f"adapt_transformers_to_gaudi failed: {e}")

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    from cosyvoice.utils.file_utils import load_wav, logging
    from cosyvoice.utils.common import set_all_random_seed

    logging.getLogger('numba').setLevel(logging.WARNING)
    torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)


    # Load CosyVoice model on HPU
    try:
        try:
            cosyvoice = CosyVoice(args.model_dir)
        except Exception:
            cosyvoice = CosyVoice2(args.model_dir)

        # Wrap key modules in HPU graphs
        device = 'hpu'
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = cosyvoice.model.llm.llm.model.bfloat16().eval().to(device)
        cosyvoice.model.llm.llm.model = wrap_in_hpu_graph(model)
        model = cosyvoice.model.llm.llm_decoder.bfloat16().eval().to(device)
        cosyvoice.model.llm.llm_decoder = wrap_in_hpu_graph(model)
        cosyvoice.model.flow = cosyvoice.model.flow.bfloat16().eval()

        sft_spk = cosyvoice.list_available_spks()
        if len(sft_spk) == 0:
            sft_spk = ['']

        # Signal readiness with runtime info
        result_queue.put({
            "type": "ready",
            "ok": True,
            "sample_rate": cosyvoice.sample_rate,
            "sft_spk": sft_spk,
        })
        ready_event.set()
    except Exception as e:
        result_queue.put({"type": "ready", "ok": False, "error": f"Model load failed: {e}"})
        ready_event.set()
        return

    # Serve tasks
    while not shutdown_event.is_set():
        task_event.wait(timeout=2)
        if shutdown_event.is_set():
            break
        task_event.clear()
        while True:
            try:
                task = task_queue.get_nowait()
            except Exception:
                break
            if task is None:
                continue

            cmd = task.get("cmd")
            if cmd == "warmup":
                try:
                    tts_text = task.get("tts_text", "你好")
                    prompt_text = task.get("prompt_text", "你好")
                    prompt_wav = task.get("prompt_wav")
                    prompt_speech_16k = load_wav(prompt_wav, 16000) if prompt_wav else None
                    set_all_random_seed(0)
                    for _ in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
                        pass
                    result_queue.put({"type": "warmup", "ok": True})
                except Exception as e:
                    result_queue.put({"type": "warmup", "ok": False, "error": str(e)})
            elif cmd == "generate":
                try:
                    tts_text = task["tts_text"]
                    mode = task["mode"]
                    sft_dropdown = task.get("sft_dropdown", '')
                    prompt_text = task.get("prompt_text", '')
                    prompt_wav = task.get("prompt_wav")
                    instruct_text = task.get("instruct_text", '')
                    seed = task.get("seed", 0)
                    stream = task.get("stream", False)
                    speed = task.get("speed", 1.0)

                    def postprocess(speech, top_db=60, hop_length=220, win_length=440):
                        import librosa
                        speech, _ = librosa.effects.trim(
                            speech, top_db=top_db,
                            frame_length=win_length,
                            hop_length=hop_length
                        )
                        if speech.abs().max() > 0.8:
                            speech = speech / speech.abs().max() * 0.8
                        speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
                        return speech

                    set_all_random_seed(seed)
                    wa_data = np.zeros(1)

                    if mode == '预训练音色':
                        gen_iter = cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=sft_dropdown, stream=stream, speed=speed)
                    elif mode == '极速复刻':
                        prompt_speech_16k = postprocess(load_wav(prompt_wav, 16000))
                        gen_iter = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed)
                    elif mode == '跨语种复刻':
                        prompt_speech_16k = postprocess(load_wav(prompt_wav, 16000))
                        gen_iter = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed)
                    else:
                        prompt_speech_16k = load_wav(prompt_wav, 16000)
                        gen_iter = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed)

                    for i in gen_iter:
                        result_queue.put({
                            "type": "chunk",
                            "ok": True,
                            "data": i['tts_speech'].numpy().flatten(),
                        })
                        if not stream:
                            result_queue.put({"type": "chunk", "ok": True, "data": wa_data})
                    result_queue.put({"type": "done", "ok": True})
                except Exception as e:
                    result_queue.put({"type": "error", "ok": False, "error": str(e)})


def _ensure_hpu_subprocess_started(args):
    global _task_queue, _result_queue, _task_event, _shutdown_event, _ready_event, _hpu_ps, _sample_rate, _sft_spk_list
    if _hpu_ps and _hpu_ps.is_alive():
        return
    _task_queue = multiprocessing.Queue(maxsize=4)
    _result_queue = multiprocessing.Queue(maxsize=32)
    _task_event = multiprocessing.Event()
    _shutdown_event = multiprocessing.Event()
    _ready_event = multiprocessing.Event()
    _hpu_ps = multiprocessing.Process(
        target=_hpu_child_process,
        args=(_task_queue, _result_queue, _task_event, _shutdown_event, _ready_event, args),
        daemon=True,
    )
    _hpu_ps.start()
    # Wait for readiness signal
    _ready_event.wait(timeout=3600)
    # Read the ready payload
    try:
        msg = _result_queue.get(timeout=10)
        if msg.get("type") == "ready" and msg.get("ok"):
            _sample_rate = msg.get("sample_rate", _sample_rate)
            _sft_spk_list = msg.get("sft_spk", _sft_spk_list)
        else:
            raise RuntimeError(msg.get("error", "HPU child failed to start"))
    except Exception as e:
        raise RuntimeError(f"Failed to start HPU child: {e}")


def _stop_hpu_subprocess():
    global _hpu_ps, _shutdown_event
    if _hpu_ps:
        _shutdown_event.set()
        _hpu_ps.join(timeout=5)
        if _hpu_ps.is_alive():
            _hpu_ps.terminate()
            _hpu_ps.join()


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    # Use child-provided sample_rate
    speech = torch.concat([speech, torch.zeros(1, int(_sample_rate * 0.2))], dim=1)
    return speech


def change_mode(mode_checkbox_group):
    instr = instruct_dict[mode_checkbox_group]
    show_instruct = True
    show_prompt_text = True
    show_prompt_audio = True
    show_sft = True

    if mode_checkbox_group == '极速复刻':
        show_instruct = False
        show_prompt_text = True
        show_prompt_audio = True
        show_sft = False
    elif mode_checkbox_group == '跨语种复刻':
        show_instruct = False
        show_prompt_text = False
        show_prompt_audio = True
        show_sft = False
    elif mode_checkbox_group == '自然语言控制':
        show_instruct = True
        show_prompt_text = False
        show_prompt_audio = True
        show_sft = False
    elif mode_checkbox_group == '预训练音色':
        show_instruct = False
        show_prompt_text = False
        show_prompt_audio = False
        show_sft = True

    # Debug: log mode and visibility flags
    # try:
    #    logging.info(f"[change_mode] mode={mode_checkbox_group} vis: sft={show_sft}, audio={show_prompt_audio}, text={show_prompt_text}, instruct={show_instruct}")
    # except Exception:
    #    pass

    return (
        instr,
        gr.update(visible=show_sft),
        gr.update(visible=show_prompt_audio),
        gr.update(visible=show_prompt_audio),
        gr.update(visible=show_prompt_text),
        gr.update(visible=show_instruct),
    )


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (_sample_rate, default_data)
        if prompt_wav is None:
            gr.Warning('您正在使用自然语言控制模式, 请提供prompt音频')
            yield (_sample_rate, default_data)
        if prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (_sample_rate, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (_sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (_sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色！')
            yield (_sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (_sample_rate, default_data)
        if instruct_text != '':
            gr.Info('您正在使用极速复刻模式，预训练音色/instruct文本会被忽略！')

    # audio_output from gr.Audio() works in streaming mode. Seems gr.Audion() in streaming mode
    # requies 2+ chunks of data. So we need to yield a dummy chunk first when stream is False
    # Delegate generation to HPU child process and stream results
    wa_data = np.zeros(1)
    prompt_wav = prompt_wav_upload or prompt_wav_record
    _task_queue.put({
        "cmd": "generate",
        "tts_text": tts_text,
        "mode": mode_checkbox_group,
        "sft_dropdown": sft_dropdown,
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav,
        "instruct_text": instruct_text,
        "seed": seed,
        "stream": stream,
        "speed": speed,
    })
    _task_event.set()
    while True:
        msg = _result_queue.get()
        mtype = msg.get("type")
        if mtype == "chunk" and msg.get("ok"):
            yield (_sample_rate, msg["data"])
        elif mtype == "done":
            break
        elif mtype == "error":
            gr.Warning(f"生成失败: {msg.get('error')}")
            yield (_sample_rate, default_data)
            break

def warmup_audio(tts_text, prompt_text, prompt_wav):
    # Request a one-shot warmup on child
    _task_queue.put({
        "cmd": "warmup",
        "tts_text": tts_text,
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav,
    })
    _task_event.set()
    # Optionally consume the warmup ack to avoid clogging the queue
    try:
        msg = _result_queue.get(timeout=30)
        # ignore content
    except Exception:
        pass

def main():
    cv_examples=[
        [
            "梯度是一个多变量微积分中的概念，用于描述一个标量场在某一点处的最大变化率，以及变化最快的方向。在物理学中，梯度通常用来表示某个物理量的空间变化情况。",
            inference_mode_list[0],
            None,
            "./asset/ZH_2_prompt.wav",
            "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
            "",
        ],
        [
            "If one knows how to be grateful and content with small things, then he is a happy person.",
            inference_mode_list[1],
            None,
            "./asset/9_ZH.wav",
            "如果能对小事感到感激和满足，那他就是幸福的人。",
            "",
        ],
        [
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            inference_mode_list[2],
            None,
            "./asset/zero_shot_prompt.wav",
            "",
            "用四川话说这句话",
        ],
        [
            "这次机会让我能够在新的领域中不断学习和成长，同时也激励我去克服自身的不足。",
            inference_mode_list[3],
            _sft_spk_list[0],
            None,
            "",
            "",
        ],
    ]
    with gr.Blocks() as demo:
        gr.Markdown("""
    <div style="text-align:center;">
      <h2><b>CosyVoice2 语音合成服务</b></h2>
    </div>

    """)
        gr.Markdown("请输入需要合成的文本，选择合成模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value=cv_examples[0][0])
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择合成模式', value=cv_examples[0][1])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[cv_examples[0][1]], scale=0.5)
            # 恢复基于初始示例的可见性，确保启动即符合预期
            init_mode = cv_examples[0][1]
            sft_dropdown = gr.Dropdown(choices=_sft_spk_list, label='选择预训练音色', value=cv_examples[0][2], scale=0.25,
                                       visible=True if init_mode == '预训练音色' else False)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1], visible=False)
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1,visible=False)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择参考音频文件，注意采样率不低于16khz', value=cv_examples[0][3],
                                         visible=False if init_mode == '预训练音色' else True)
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制参考音频文件',
                                         visible=False if init_mode == '预训练音色' else True)
        prompt_text = gr.Textbox(label="输入参考文本", lines=1, placeholder="请输入参考文本，需与参考音频内容一致，暂时不支持自动识别...", value=cv_examples[0][4],
                     visible=True if init_mode == '极速复刻' else False)
        instruct_text = gr.Textbox(label="输入控制文本", lines=1, placeholder="请输入控制文本，比如: 用四川话说这句话", value=cv_examples[0][5],
                       visible=True if init_mode == '自然语言控制' else False)

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        # Examples: set radio value and update visibility even on same-mode reselection
        def apply_example_and_update(tts_text_val, mode_val, sft_val, wav_upload_val, prompt_text_val, instruct_text_val):
            """Workaround: force a mode toggle to ensure visibility updates on reselection.
            First yield a temporary '极速复刻' state, then yield the target mode state.
            """
            try:
                logging.info(f"[examples] requested mode={mode_val}; toggling via '极速复刻' then back")
            except Exception:
                pass

            # 1) Yield a temporary state to force Gradio to treat inputs as changed
            temp_mode = '极速复刻'
            temp_instr, temp_sft, temp_upload, temp_record, temp_ptext, temp_instruct = change_mode(temp_mode)
            yield (
                gr.update(value=temp_mode),
                temp_instr,
                temp_sft,
                temp_upload,
                temp_record,
                temp_ptext,
                temp_instruct,
            )

            # 2) Yield the final desired state
            instr, sft_vis, upload_vis, record_vis, ptext_vis, instruct_vis = change_mode(mode_val)
            yield (
                gr.update(value=mode_val),
                instr,
                sft_vis,
                upload_vis,
                record_vis,
                ptext_vis,
                instruct_vis,
            )

        gr.Examples(
            fn=apply_example_and_update,
            examples=cv_examples,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_wav_upload, prompt_text, instruct_text],
            outputs=[mode_checkbox_group, instruction_text, sft_dropdown, prompt_wav_upload, prompt_wav_record, prompt_text, instruct_text],
            cache_examples=False,
            run_on_click=True,
        )
        # Restore radio change to update visibility on manual selection
        mode_checkbox_group.change(
            fn=change_mode,
            inputs=[mode_checkbox_group],
            outputs=[instruction_text, sft_dropdown, prompt_wav_upload, prompt_wav_record, prompt_text, instruct_text],
        )
        # 在启动时应用一次可见性同步，确保初始状态与模式一致
        demo.load(
            fn=change_mode,
            inputs=[mode_checkbox_group],
            outputs=[instruction_text, sft_dropdown, prompt_wav_upload, prompt_wav_record, prompt_text, instruct_text],
        )
        # Radio change alone governs visibility to avoid races.

    warmup_audio(cv_examples[1][0], cv_examples[1][4], cv_examples[1][3])
    demo.queue(max_size=32, default_concurrency_limit=1)
    demo.launch(server_name='0.0.0.0', server_port=args.port,
                ssl_certfile="./cert.pem", ssl_keyfile="./key.pem", ssl_verify=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='/workspace/data/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()

    # Initialize child process which loads model and provides runtime info
    _ensure_hpu_subprocess_started(args)

    # Use child-provided runtime values
    prompt_sr = 16000
    default_data = np.zeros(_sample_rate)

    try:
        main()
    finally:
        _stop_hpu_subprocess()
