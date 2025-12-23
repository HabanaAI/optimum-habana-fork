# Prepare model

```shell
cd CosyVoice
apt install git-lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
cp spk2info.pt pretrained_models/CosyVoice2-0.5B/
```


# Install
```shell
cd third_party/Matcha-TTS
pip install -e .
cd ../..
pip install -r requirements-hpu.txt
pip install opea-comps==1.3
```


# Simple Commad Demo
```shell
python hpu_cosyvoice_test.py
```

# API server
## API Description
| Parameter Name       | Type          | Default    | Description                                        |
| -------------------- | ------------- | ---------- | -------------------------------------------------- |
| text                 | str           | None       | Text for generating audio
| mode                 | str           | None       | Method for generating audio, options: zero_shot, cross_lingual, instruct, pretrain
| pretrained_tone      | str           | None       | Pre-trained tone
| prompt_text          | str           | None       | Text of reference audio
| prompt_audio         | file          | None       | Reference audio
| instruct_text        | str           | None       | Text for natural language control of audio generation
| model                | str           | None       | Model name
| speed                | float         | 1.0        | Rate of generated audio
| seed                 | int           | 0          | Random seed for generation

## Return Value Description
| Parameter Name       | Type          | Description                                        |
| -------------------- | ------------- | -------------------------------------------------- |
| id                   | str           | Task ID, used for querying and retrieving results
| model                | str           | Model name
| status               | str           | Task status, queued, progressing, completed, deleted, error
| progress             | int           | Task progress 0-100 (currently only 0, 100 are possible)
| created_time         | str           | Task creation time
| started_time         | str           | Task start time
| finished_time        | str           | Task completion time
| queue_length         | int           | Number of tasks ahead in queue
| error                | str           | Error message

## Command Description
## zero_shot
Generate audio by replicating the tone of prompt_audio based on text. prompt_audio and prompt_text need to match in content.

Required parameters: text, mode, prompt_text, prompt_audio

Command example:
```shell
curl http://10.239.15.29:9370/v1/audio/speech \
    -F text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
    -F mode="zero_shot" \
    -F prompt_text="希望你以后能够做的比我还好呦。" \
    -F prompt_audio="@asset/zero_shot_prompt.wav"
```

Output example:
```shell
{"id":"379b56e2-dafe-11f","model":"iic/CosyVoice2-0.5B","status":"queued","progress":0,"created_time":"2025-12-17 12:09:43","started_time":"","finished_time":"","queue_length":1,"error":""}
```

## cross_lingual
Generate audio by replicating the tone of prompt_video based on text across languages

Required parameters: text, mode, prompt_text, prompt_audio

Command example:
```shell
curl http://10.239.15.29:9370/v1/audio/speech \
    -F text="If one knows how to be grateful and content with small things, then he is a happy person." \
    -F mode="cross_lingual" \
    -F prompt_audio="@asset/cross_lingual_prompt.wav"
```

Output example:
```shell
{"id":"75f541ae-daff-11f","model":"iic/CosyVoice2-0.5B","status":"queued","progress":0,"created_time":"2025-12-17 12:18:37","started_time":"","finished_time":"","queue_length":1,"error":""}
```

## instruct
Generate audio related to text based on instruct_text, with prompt_audio providing tone reference

Required parameters: text, mode, instruct_text, prompt_audio

Command example:
```shell
curl http://10.239.15.29:9370/v1/audio/speech \
    -F text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
    -F mode="instruct" \
    -F instruct_text="用四川话说这句话" \
    -F prompt_audio="@asset/zero_shot_prompt.wav"
```

Output example:
```shell
{"id":"46308b8a-db00-11f","model":"iic/CosyVoice2-0.5B","status":"queued","progress":0,"created_time":"2025-12-17 12:24:27","started_time":"","finished_time":"","queue_length":1,"error":""}
```

## pretrain
Generate audio using the pre-trained tone pretrained_tone

Required parameters: text, mode, pretrained_tone

Command example:
```shell
curl http://10.239.15.29:9370/v1/audio/speech \
    -F text="这次机会让我能够在新的领域中不断学习和成长，同时也激励我去克服自身的不足。" \
    -F mode="pretrain" \
    -F pretrained_tone="Chinese Male"
```

Output example:
```shell
{"id":"82b9ea88-db00-11f","model":"iic/CosyVoice2-0.5B","status":"queued","progress":0,"created_time":"2025-12-17 12:26:08","started_time":"","finished_time":"","queue_length":1,"error":""}
```

### Query Available Tones
```shell
curl http://10.239.15.29:9370/v1/audio/speech/query/pretrained_tone
```

Output example:
```shell
{"success":{"message":"['Chinese Female', 'Chinese Male', 'Japanese Male', 'Cantonese Female', 'English Female', 'English Male', 'Korean Female']","code":"200"}}
```

## Query Task Status
```shell
curl http://10.239.15.29:9370/v1/audio/speech/46308b8a-db00-11f
```

Output example:
```shell
{"id":"46308b8a-db00-11f","model":"iic/CosyVoice2-0.5B","status":"completed","progress":100,"created_time":"2025-12-17 12:24:27","started_time":"2025-12-17 12:24:27","finished_time":"2025-12-17 12:25:02","queue_length":0,"error":""}
```

## Download Audio
```shell
curl http://10.239.15.29:9370/v1/audio/speech/46308b8a-db00-11f/content -o test.wav
```

## Delete Task
Only tasks in queued status can be deleted

Command example:
```shell
curl http://10.239.15.29:9370/v1/audio/speech/5da20842-db01-11f/delete
```

Output example:
```shell
{"success":{"message":"task 5da20842-db01-11f is deleted","code":"200"}}

