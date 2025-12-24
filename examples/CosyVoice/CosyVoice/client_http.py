import asyncio
import aiohttp
import json
import base64
import uuid
import time

API_URL = "http://localhost:8022/v1/audio/speech"

CONCURRENT_REQUESTS = 10  # 并发数量
# 请求 payload 示例
payload_template = {
    # 输入提示词长度
    "input": "和你聊天真的很开心",
    # "input": "云主机是一种按需获取的云端服务器，为您提供高可靠、弹性扩展的计算资源服务，您可以根据需求选择不同规格的CPU、内存、操作系统、硬盘和网络来创建您的云主机，满足您的个性化业务需求。云主机从订购到使用仅需数十秒时间，助您快速灵活地构建企业应用。",
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "voicepath": "/CosyVoice/asset/zero_shot_prompt.wav",
    "voice": "中文女",
    "response_format": "wav",
    "sample_rate": 24000,
    "stream": False,
    "speed": 1
}
async def send_request(session: aiohttp.ClientSession, idx: int):
    try:
        start_time = time.time()
        async with session.post(API_URL, json=payload_template) as response:
            if response.status == 200:
                audio_bytes = await response.read()
                end_time = time.time()
                print(f"infer time cost: {(end_time - start_time)}")

                filename = f"22output_zero_{idx}.wav"
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                end_time2 = time.time()
                print(f"[{idx}] ✅ Saved to {filename}")
                print(f"infer time cost: {(end_time2 - start_time)}")
            else:
                text = await response.text()
                print(f"[{idx}] ❌ Error {response.status}: {text}")
    except Exception as e:
        print(f"[{idx}] ❌ Exception: {e}")

async def main():
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(CONCURRENT_REQUESTS)]
        await asyncio.gather(*tasks)
    all_time = time.time()-start_time
    print(f"{CONCURRENT_REQUESTS}并发总耗时：{all_time}" )

if __name__ == "__main__":
    asyncio.run(main())
