# Install
```shell
cd SongGeneration
wget https://media.githubusercontent.com/media/tencent-ailab/SongGeneration/refs/heads/main/tools/new_prompt.pt
mv new_prompt.pt tools/
pip install -r requirements-gaudi.txt
# Note, if any conflict issue ,please try to use opea-comps==1.5
pip install opea-comps==1.3 
huggingface-cli download lglg666/SongGeneration-Runtime --local-dir ./runtime
mv runtime/ckpt ckpt
mv runtime/third_party third_party
apt update
apt install ffmpeg -y
```


# Prepare model

```shell
# download SongGeneration-base
huggingface-cli download lglg666/SongGeneration-base --local-dir ./songgeneration_base
# download SongGeneration-base-new
huggingface-cli download lglg666/SongGeneration-base-new --local-dir ./songgeneration_base_new
# download SongGeneration-base-full
huggingface-cli download lglg666/SongGeneration-base-full --local-dir ./songgeneration_base_full
# download SongGeneration-large
huggingface-cli download lglg666/SongGeneration-large --local-dir ./songgeneration_large
```


# Simple Commad Demo
```shell
PT_HPU_LAZY_MODE=1 ./generate.sh songgeneration_base_full sample/lyrics.jsonl sample/output --not_use_flash_attn
```

# API server
```shell
bash api_server.sh
```

# API server
## API Description
| Parameter Name        | Type          | Default    | Description                                        |
| --------------------  | ------------- | ---------- | -------------------------------------------------- |
| gt_lyric              | str           | None       | Defines the lyrics and structure of the song
| descriptions          | str           | None       | Allows you to control various musical attributes of the generated song
| prompt_audio          | file          | None       | Influence genre, instrumentation, rhythm, and voice
| auto_prompt_audio_type| str           | None       | Automatic reference selection
| gen_type              | str           | None       | "separate" or "mixed". use "separate" to also generate bgm and vocal
| model                 | str           | None       | Model name
| seed                  | int           | 0          | Random seed for generation

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

## Input details
### 🎵 Lyrics Input Format

The `gt_lyric` field defines the lyrics and structure of the song. It consists of multiple musical section, each starting with a structure label. The model uses these labels to guide the musical and lyrical progression of the generated song.

#### 📌 Structure Labels

- The following segments **should not** contain lyrics (they are purely instrumental):

  - `[intro-short]`, `[intro-medium]`, `[inst-short]`, `[inst-medium]`, `[outro-short]`, `[outro-medium]`

  > - `short` indicates a segment of approximately 0–10 seconds
  > - `medium` indicates a segment of approximately 10–20 seconds
  > - We find that [inst] label is less stable, so we recommend that you do not use it.

- The following segments **require lyrics**:

  - `[verse]`, `[chorus]`, `[bridge]`

#### 🧾 Lyrics Formatting Rules

- Each section is **separated by ` ; `**

- Within lyrical segments (`[verse]`, `[chorus]`, `[bridge]`), lyrics must be written in complete sentences and separated by a period (`.`)

- A complete lyric string may look like:

  ```
  [intro-short] ; [verse] These faded memories of us. I can't erase the tears you cried before. Unchained this heart to find its way. My peace won't beg you to stay ; [bridge] If ever your truth still remains. Turn around and see. Life rearranged its games. All these lessons in mistakes. Even years may never erase ; [inst-short] ; [chorus] Like a fool begs for supper. I find myself waiting for her. Only to find the broken pieces of my heart. That was needed for my soul to love again ; [outro-short]
  ```

- More examples can be found in `sample/test_en_input.jsonl` and `sample/test_zh_input.jsonl`.

### 📝 Description Input Format

The `descriptions` field allows you to control various musical attributes of the generated song. It can describe up to six musical dimensions: **Gender** (e.g., male, female), **Timbre** (e.g., dark, bright, soft), **Genre** (e.g., pop, jazz, rock), **Emotion** (e.g., sad, energetic, romantic), **Instrument** (e.g., piano, drums, guitar), **BPM** (e.g., the bpm is 120). 

- All six dimensions are optional — you can specify any subset of them.

- The order of dimensions is flexible.

- Use **commas (`,`)** to separate different attributes.

- Although the model supports open vocabulary, we recommend using predefined tags for more stable and reliable performance. A list of commonly supported tags for each dimension is available in the `sample/description/` folder.

- Here are a few valid `descriptions` inputs:

  ```
  - female, dark, pop, sad, piano and drums.
  - male, piano, jazz.
  - male, dark, the bpm is 110.
  ```

### 🎧Prompt Audio Usage Notes

- The input audio file can be longer than 10 seconds, but only the first 10 seconds will be used.
- For best musicality and structure, it is recommended to use the chorus section of a song as the prompt audio.
- You can use this field to influence genre, instrumentation, rhythm, and voice

#### ⚠️ Important Considerations

- **Avoid providing both `prompt_audio_path` and `descriptions` at the same time.**
  If both are present, and they convey conflicting information, the model may struggle to follow instructions accurately, resulting in degraded generation quality.
- If `prompt_audio_path` is not provided, you can instead use `auto_prompt_audio_type` for automatic reference selection.



## Command Description
## send request
Command example:
```shell
curl http://10.239.15.29:8484/v1/audio/song \
    --form-string gt_lyric="[intro-long] ; [verse] 花朵绽放如诗篇.随风轻舞动.湖面波光映倒影.心随波浪荡漾中.感受着这美好时光.仿佛梦境般飘渺 ; [chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅.阳光洒满湖面.一切都如此美好.仿佛置身人间天堂.让人心醉神迷 ; [inst-long] ; [verse] 柳枝轻拂水面.带来淡淡的花香.渔夫划桨而过.留下一道道涟漪.在这宁静的午后.一切都显得如此和谐 ; [chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮.雨滴轻敲窗棂.夜风中烛火摇曳.在这人间天堂.让我们紧紧相拥 ; [outro-long]"
```

```shell
curl http://10.239.15.29:8484/v1/audio/song \
    --form-string gt_lyric="[intro-long] ; [verse] 花朵绽放如诗篇.随风轻舞动.湖面波光映倒影.心随波浪荡漾中.感受着这美好时光.仿佛梦境般飘渺 ; [chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅.阳光洒满湖面.一切都如此美好.仿佛置身人间天堂.让人心醉神迷 ; [inst-long] ; [verse] 柳枝轻拂水面.带来淡淡的花香.渔夫划桨而过.留下一道道涟漪.在这宁静的午后.一切都显得如此和谐 ; [chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮.雨滴轻敲窗棂.夜风中烛火摇曳.在这人间天堂.让我们紧紧相拥 ; [outro-long]", \
    -F descriptions="female, dark, sad, piano and drums"
```

```shell
curl http://10.239.15.29:8484/v1/audio/song \
    --form-string gt_lyric="[intro-long] ; [verse] 花朵绽放如诗篇.随风轻舞动.湖面波光映倒影.心随波浪荡漾中.感受着这美好时光.仿佛梦境般飘渺 ; [chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅.阳光洒满湖面.一切都如此美好.仿佛置身人间天堂.让人心醉神迷 ; [inst-long] ; [verse] 柳枝轻拂水面.带来淡淡的花香.渔夫划桨而过.留下一道道涟漪.在这宁静的午后.一切都显得如此和谐 ; [chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮.雨滴轻敲窗棂.夜风中烛火摇曳.在这人间天堂.让我们紧紧相拥 ; [outro-long]", \
    -F auto_prompt_audio_type="Metal"
```

```shell
curl http://10.239.15.29:8484/v1/audio/song \
    --form-string gt_lyric="[intro-long] ; [verse] 花朵绽放如诗篇.随风轻舞动.湖面波光映倒影.心随波浪荡漾中.感受着这美好时光.仿佛梦境般飘渺 ; [chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅.阳光洒满湖面.一切都如此美好.仿佛置身人间天堂.让人心醉神迷 ; [inst-long] ; [verse] 柳枝轻拂水面.带来淡淡的花香.渔夫划桨而过.留下一道道涟漪.在这宁静的午后.一切都显得如此和谐 ; [chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮.雨滴轻敲窗棂.夜风中烛火摇曳.在这人间天堂.让我们紧紧相拥 ; [outro-long]", \
    -F prompt_audio="@sample/sample_prompt_audio.wav"
```

```shell
curl http://10.239.15.29:8484/v1/audio/song \
    --form-string gt_lyric="[intro-long] ; [verse] 花朵绽放如诗篇.随风轻舞动.湖面波光映倒影.心随波浪荡漾中.感受着这美好时光.仿佛梦境般飘渺 ; [chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅.阳光洒满湖面.一切都如此美好.仿佛置身人间天堂.让人心醉神迷 ; [inst-long] ; [verse] 柳枝轻拂水面.带来淡淡的花香.渔夫划桨而过.留下一道道涟漪.在这宁静的午后.一切都显得如此和谐 ; [chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮.雨滴轻敲窗棂.夜风中烛火摇曳.在这人间天堂.让我们紧紧相拥 ; [outro-long]", \
    -F gen_type="separate"
```

Output example:
```shell
{"id":"379b56e2-dafe-11f","model":"iic/CosyVoice2-0.5B","status":"queued","progress":0,"created_time":"2025-12-17 12:09:43","started_time":"","finished_time":"","queue_length":1,"error":""}
```

### Query auto_prompt_audio_type
```shell
curl http://10.239.15.29:8484/v1/audio/song/query/auto_prompt_audio_type
```

Output example:
```shell
{"success":{"message":"['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']","code":"200"}}
```

## Query Task Status
```shell
curl http://10.239.15.29:8484/v1/audio/song/2f1655d0-f5b6-11f
```

Output example:
```shell
{"id":"2f1655d0-f5b6-11f","model":"SongGeneration","status":"queued","progress":0,"created_time":"2026-01-20 12:12:07","started_time":"","finished_time":"","queue_length":5,"error":""}
```

## Download Audio
```shell
curl http://10.239.15.29:8484/v1/audio/song/2f1655d0-f5b6-11f/content -o test.wav
```

## Download BGM Audio
```shell
curl http://10.239.15.29:8484/v1/audio/song/2f1655d0-f5b6-11f/bgm/content -o test.wav
```

## Download Vocal Audio
```shell
curl http://10.239.15.29:8484/v1/audio/song/2f1655d0-f5b6-11f/vocal/content -o test.wav
```

## Delete Task
Only tasks in queued status can be deleted

Command example:
```shell
curl http://10.239.15.29:8484/v1/audio/song/2f1655d0-f5b6-11f/delete
```

Output example:
```shell
{"success":{"message":"task 2f1655d0-f5b6-11f is deleted","code":"200"}}
```
