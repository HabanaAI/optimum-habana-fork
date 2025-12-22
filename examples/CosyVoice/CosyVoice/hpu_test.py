import os
# lazy mode
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
from cosyvoice.utils.file_utils import load_wav

cosyvoice = CosyVoice2('/workspace/data/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

#print(cosyvoice.model.flow.decoder.estimator.down_blocks)
#exit()
torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

nwarmup = 1
loop = 3
device='hpu'
model = cosyvoice.model.llm.llm.model.bfloat16().eval().to(device)
cosyvoice.model.llm.llm.model = wrap_in_hpu_graph(model)

model = cosyvoice.model.llm.llm_decoder.bfloat16().eval().to(device)
cosyvoice.model.llm.llm_decoder = wrap_in_hpu_graph(model)

cosyvoice.model.flow = cosyvoice.model.flow.bfloat16().eval()#.to(device)
#cosyvoice.model.flow = wrap_in_hpu_graph(model)

#model = cosyvoice.model.hift.eval().to(device)
#cosyvoice.model.hift = model #wrap_in_hpu_graph(model)

t1 = time.time()
with torch.no_grad():
    for _ in range(nwarmup):
        for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的>祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
            continue
t2 = time.time()
t = (t2 - t1) * 1000.0
print("Warmup Latency: {:.3f} ms".format(t))

tts_str = "《天道》是一部集爱情、商战于一身的电视剧，涉及到政治、商战、爱情等诸多方面，是一部比较另类的作品，是一部电视剧史上从未出现过的电视剧。是一部发烧友必看的活教材，而它所描述的商人之间的尔虞我诈、勾心斗角、尤其是商界怪才丁元英那不按常规出牌的商人手腕，又可以让众多商人学到许多东西，因此，《天道》又被称为商人必看的教科书。一位资深业界人士指出，这是一部外行看热闹，内行看门道，女人看爱情，商人看商战的好戏，不同的人可以从中找出自己不同的东西，可以领受到不同的感悟。丁元英的私募基金是一家以德国几家金融公司为资本委托方的边缘公司，在中国股市进行了11个月的掠夺式经营之后，作为一个中国人，他对掠夺式的股市操作心里不堪重负，充满了矛盾与无奈。他以“个人心理状态”为由中止了私墓基金的合作，他交代助理肖亚文，在北京附近的城市租一套房子，他要远离大都市的喧闹，找个僻静地方一个人清静一段时间。肖亚文是个非常有头脑的白领女子，而她需要与丁元英保持一定联系，因为丁元英有着与正常人完全颠倒的思维，认识这个人就意味着给自己的思想、观念开了一扇窗户，能让她思考、觉悟，甚至将来可能的机会、帮助。肖亚文小题大做的从北京飞抵德国法兰克福，求助于正在法兰克福探亲的警官大学同窗好友、古城公安局刑警芮小丹，请她帮忙在古城租一套房子，芮小丹了解了肖亚文真正意图之后，理解了肖亚文貌似夸张的做法，并答应了她的要求，却让芮小丹对这个从未见过面的男人有了一种先入为主的反感。　　丁元英到古城后一直过着与任何人没有来往的平静日子，8个月时间过去了，因为缺少生活费，丁元英将自己收藏的唱片拿到刘冰的“孤岛唱片店”去变卖，临近春节的时候芮小丹想起了这个几乎在她记忆里已经不存在的人，考虑到他在古城的“暂住证”和预交的房租都到期了，她给丁元英打了一个电话，并去看了他，无意中听到了丁元英的音响，她被那种纯美的音乐打动了，她向丁元英询问这套音响的价格，丁元英只能含糊地说“得几万吧”。　　芮小丹开着警车在古城各个音响店寻找与丁元英同样的音响，因此而影响了工作，受到了通报批评和停职反省处理。丁元英对音响价格的含糊表态和变卖唱片的窘迫处境使芮小丹既有尴尬的恼羞成怒，又有愧对朋友所托的内疚。芮小丹请丁元英出来吃饭，想让丁元英喝醉以后出丑，席间，芮小丹被丁元英的学识和气度所折服，欧阳雪察觉到了芮小丹的变化。　　确定了自己的感情之后"

with torch.no_grad(): #, torch.autocast(device_type="hpu", dtype=torch.bfloat16):
    for _ in range(loop):
        t1 = time.time()
        for i, j in enumerate(cosyvoice.inference_instruct2(tts_str, '用四川话说这句话', prompt_speech_16k, stream=False)):
            torchaudio.save('/workspace/output/cosyvoice.inference/instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
#        for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#            torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
#        for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#            torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        t2 = time.time()
        t = (t2 - t1) * 1000.0
        print("LLM Infer Latency: {:.3f} ms".format(t))

