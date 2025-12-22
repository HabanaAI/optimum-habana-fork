
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
     #   "　　丁元英到古城后一直过着与任何人没有来往的平静日子，八个月时间过去了，因为缺少生活费，丁元英将自己收藏的唱片拿到刘冰的“孤岛唱片店”去变卖，临近春 节的时候芮小丹想起了这个几乎在她记忆里已经不存在的人，考虑到他在古城的“暂住证”和预交的房租都到期了，她给丁元英打了一个电话，并去看了他，无意中听到了丁元英的音响，她被那种纯美的音乐打动了，她 向丁元英询问这套音响的价格，丁元英只能含糊地说“得几万吧”。",
     #   "《天道》是一部集爱情、商战于一身的电视剧，涉及到政治、商战、爱情等诸多方面，是一部比较另类的作品，是一部电视剧史上从未出现过的电视剧。是一部发烧友必看的活教材，而它所描述的商人之间的尔虞我诈、勾心斗角、尤其是商界怪才丁元英那不按常规出牌的商人手腕，又可以让众多商人学到许多东西，因此，《天道》又被称为商人必看的教科书。一位资深业界人士指出，这是一部外行看热闹，内行看门道，女人看爱情，商人看商战的好戏，不同的人可以从中找出自己不同的东西，可以领受到不同的感悟。丁元英的私募基金是一家以德国几家金融公司为资本委托方的边缘公司，在中国股市进行了11个月的掠夺式经营之后，作为一个中国人，他对掠夺式的股市操作心里不堪重负，充满了矛盾与无奈。他以“个人心理状态”为由中止了私墓基金的合作，他交代助理肖亚文，在北京附近的城市租一套房子，他要远离大都市的喧闹，找个僻静地方一个人清静一段时间。肖亚文是个非常有头脑的白领女子，而她需要与丁元英保持一定联系，因为丁元英有着与正常人完全颠倒的思维，认识这个人就意味着给自己的思想、观念开了一扇窗户，能让她思考、觉悟，甚至将来可能的机会、帮助。肖亚文小题大做",
     #   "《天道》是一部集爱情、商战于一身的电视剧，涉及到政治、商战、爱情等诸多方面，是一部比较另类的作品，是一部电视剧史上从未出现过的电视剧。是一部发烧友必看的活教材，而它所描述的商人之间的尔虞我诈、勾心斗角、尤其是商界怪才丁元英那不按常规出牌的商人手腕，又可以让众多商人学到许多东西，因此，《天道》又被称为商人必看的教科书。一位资深业界人士指出，这是一部外行看热闹，内行看门道，女人看爱情，商人看商战的好戏，不同的人可以从中找出自己不同的东西，可以领受到不同的感悟。丁元英的私募基金是一家以德国几家金融公司为资本委托方的边缘公司，在中国股市进行了11个月的掠夺式经营之后，作为一个中国人，他对掠夺式的股市操作心里不堪重负，充满了矛盾与无奈。他以“个人心理状态”为由中止了私墓基金的合作，他交代助理肖亚文，在北京附近的城市租一套房子，他要远离大都市的喧闹，找个僻静地方一个人清静一段时间。肖亚文是个非常有头脑的白领女子，而她需要与丁元英保持一定联系，因为丁元英有着与正常人完全颠倒的思维，认识这个人就意味着给自己的思想、观念开了一扇窗户，能让她思考、觉悟，甚至将来可能的机会、帮助。肖亚文小题大做的从北京飞抵德国法兰克福，求助于正在法兰克福探亲的警官大学同窗好友、古城公安局刑警芮小丹，请她帮忙在古城租一套房子，芮小丹了解了肖亚文真正意图之后，理解了肖亚文貌似夸张的做法，并答应了她的要求，却让芮小丹对这个从未见过面的男人有了一种先入为主的反感。　　丁元英到古城后一直过着与任何人没有来往的平静日子，8个月时间过去了，因为缺少生活费，丁元英将自己收藏的唱片拿到刘冰的“孤岛唱片店”去变卖，临近春节的时候芮小丹想起了这个几乎在她记忆里已经不存在的人，考虑到他在古城的“暂住证”和预交的房租都到期了，她给丁元英打了一个电话，并去看了他，无意中听到了丁元英的音响，她被那种纯美的音乐打动了，她向丁元英询问这套音响的价格，丁元英只能含糊地说“得几万吧”。　　芮小丹开着警车在古城各个音响店寻找与丁元英同样的音响，因此而影响了工作，受到了通报批评和停职反省处理。丁元英对音响价格的含糊表态和变卖唱片的窘迫处境使芮小丹既有尴尬的恼羞成怒，又有愧对朋友所托的内疚。芮小丹请丁元英出来吃饭，想让丁元英喝醉以后出丑，席间，芮小丹被丁元英的学识和气度所折服，欧阳雪察觉到了芮小丹的变化。　　确定了自己的感情之后",
        "《天道》是一部集爱情、商战于一身的电视剧，涉及到政治、商战、爱情等诸多方面，是一部比较另类的作品，是一部电视剧史上从未出现过的电视剧。是一部发烧友必看的活教材，而它所描述的商人之间的尔虞我诈、勾心斗角、尤其是商界怪才丁元英那不按常规出牌的商人手腕，又可以让众多商人学到许多东西，因此，《天道》又被称为商人必看的教科书。一位资深业界人士指出，这是一部外行看热闹，内行看门道，女人看爱情，商人看商战的好戏，不同的人可以从中找出自己不同的东西，可以领受到不同的感悟。丁元英的私募基金是一家以德国几家金融公司为资本委托方的边缘公司，在中国股市进行了11个月的掠夺式经营之后，作为一个中国人，他对掠夺式的股市操作心里不堪重负，充满了矛盾与无奈。他以“个人心理状态”为由中止了私墓基金的合作，他交代助理肖亚文，在北京附近的城市租一套房子，他要远离大都市的喧闹，找个僻静地方一个人清静一段时间。肖亚文是个非常有头脑的白领女子，而她需要与丁元英保持一定联系，因为丁元英有着与正常人完全颠倒的思维，认识这个人就意味着给自己的思想、观念开了一扇窗户，能让她思考、觉悟，甚至将来可能的机会、帮助。肖亚文小题大做的从北京飞抵德国法兰克福，求助于正在法兰克福探亲的警官大学同窗好友、古城公安局刑警芮小丹，请她帮忙在古城租一套房子，芮小丹了解了肖亚文真正意图之后，理解了肖亚文貌似夸张的做法，并答应了她的要求，却让芮小丹对这个从未见过面的男人有了一种先入为主的反感。　　丁元英到古城后一直过着与任何人没有来往的平静日子，8个月时间过去了，因为缺少生活费，丁元英将自己收藏的唱片拿到刘冰的“孤岛唱片店”去变卖，临近春节的时候芮小丹想起了这个几乎在她记忆里已经不存在的人，考虑到他在古城的“暂住证”和预交的房租都到期了，她给丁元英打了一个电话，并去看了他，无意中听到了丁元英的音响，她被那种纯美的音乐打动了，她向丁元英询问这套音响的价格，丁元英只能含糊地说“得几万吧”。　　芮小丹开着警车在古城各个音响店寻找与丁元英同样的音响，因此而影响了工作，受到了通报批评和停职反省处理。丁元英对音响价格的含糊表态和变卖唱片的窘迫处境使芮小丹既有尴尬的恼羞成怒，又有愧对朋友所托的内疚。芮小丹请丁元英出来吃饭，想让丁元英喝醉以后出丑，席间，芮小丹被丁元英的学识和气度所折服，欧阳雪察觉到了芮小丹的变化。　　确定了自己的感情之后，芮小丹不计代价地为丁元英租房子、开始关心丁元英的生活。丁元英被感动了。　　发烧人士冯士杰请芮小丹去王庙村，让芮小丹亲眼看见了王庙村的贫困状况。经过思考，芮小丹决定向丁元英要一个“神话”的礼物，让他在王庙村写一个脱贫致富的神话。丁元英明知这样要求可能是个错误，然而感情的驱使却使他无法拒绝。丁元英经过反复思考，设计了一套既使古城的几个发烧友和王庙村的农户相互依存又让他们在法律上各自独立的“杀富济贫”的方案，他把目标放在了北京召开的国际音响展示会。　　格律诗公司成立了，欧阳雪、冯士杰、叶晓明和刘冰成为公司股东，丁元英告诉他们：救世主是没有的，只有自己救自己。在北京开幕的音响展示会中，丁元英以平价销售格律诗音响的策略，在当天就销售一空，此种降价行为给国内著名品牌乐圣公司造成了巨大损失，乐圣总裁林雨峰决定以《中华人民共和国反不正当竞争法》为依据起诉格律诗公司，提出诉讼要求600万元的赔偿。此事早在丁元英意料之中，他让欧阳雪去北京找肖亚文为格律诗公司代理诉讼事务。肖亚文也认为这对自己是一个机会。芮小丹目睹了格律诗公司从组建到应乐圣公司诉讼的整个过程，对于正在发生的和可以预见的这些事情，她开始思考什么是神话、什么是得救、什么是文化属性了。　　肖亚文接管格律诗公司后，没有提交应诉答辩状，放弃了答辩权利，直接进入证据交换程序，乐圣公司在北京与格律诗公司完成证据交换以后，才知道格律诗是一个扶贫公司，林雨峰意识到胜诉几乎是不可能了。他决定拼死一搏，林雨峰之所以要打这场官司，是借这场官司把丁元英这个人从幕后推到前台。　　芮小丹在办完省厅刑侦处的大案后返回县城的路上遇见被通缉的要犯黄福海、刘东昌、吴建军等人，芮小丹知道自己可能会牺牲，作为警察，她的天职就是打击犯罪，她没有避险的权利。她给丁元英打电话告别，面对这个电话，丁元英沉默了。芮小丹向分局通报情况请求支援后关了手机。一番心理较量和实战之后，吴建军自杀性爆炸死亡，芮小丹被炸残、毁容，刘东昌带着三十万现金逃跑，黄福海企图夺芮小丹的越野车，被芮小丹打伤双腿，增援人员赶到现场的时候，芮小丹开枪自杀。失去芮小丹，丁元英伤心过度吐血了。　　法院开庭宣判格律诗胜诉，林雨峰通过电视观看了法庭审理的现场直播，他开车来到盘山公路上冲下悬崖，给外界的印象是因为疲劳驾驶而发生的意外。这场诉讼在乐圣知名品牌的烘托和媒体的大肆炒作下使格律诗公司一夜之间名扬四方",
        '极速复刻',
        None,
        "./asset/ZH_2_prompt.wav",
        "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
        "",

    #    '自然语言控制',
    #    None,
    #    "./asset/zero_shot_prompt.wav",
    #    "",
    #    "用四川话说这句话",
    ],

    # [
    #     "我和你聊天真的很开心",
    #     '极速复刻',
    #     None,
    #     "./asset/ZH_2_prompt.wav",
    #     "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
    #     "",
    # ],
    # [
    #     "云主机是一种按需获取的云端服务器，为您提供高可靠、弹性扩展的计算资源服务，您可以根据需求选择不同规格的CPU、内存、操作系统、硬盘和网络来创建您的云主机，满足您的个性化业务需求。云主机从订购到使用仅需数十秒时间",
    #     '极速复刻',
    #     None,
    #     "./asset/ZH_2_prompt.wav",
    #     "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
    #     "",
    # ],
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
    # concurrency = [1, 10]
    # concurrency = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    # concurrency = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    # concurrency = [1, 50, 100, 200, 500, 1000]
    concurrency = [1]
    for case in concurrency:
        requests_per_connection = 1
        asyncio.run(main(result_file_path, case, requests_per_connection))
