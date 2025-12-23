# Text2Video 服务

OPEA Text-to-Video (文本到视频) 微服务，用于根据文本提示和音频输入生成视频。

## 概述

本项目提供 OPEA Text2Video 组件的独立部署方案。它通过 REST API 提供先进的视频生成能力，并针对英特尔 ® Habana® Gaudi® 加速器进行了优化。

## 主要特性

- **文生视频**: 支持文本提示和音频条件输入，生成动态视频。
- **任务队列管理**: 高效管理并发请求，确保服务稳定性。
- **HPU/Gaudi 优化**: 充分利用 Habana Gaudi 加速器的高性能计算能力。
- **RESTful API**: 提供标准化 RESTful 接口及 OpenAPI 类似接口。
- **容器化部署**: 支持 Docker 快速部署和环境隔离。

## 安装部署

### 1. 构建 Docker 镜像

在构建镜像前，请根据您的网络环境设置代理（如果需要）。

```bash
# 设置代理（可选）
export http_proxy="http://your-proxy-address:port"
export https_proxy="http://your-proxy-address:port"

# 执行构建命令
docker build -t text2video-gaudi:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f Dockerfile .
```

### 2. 创建 Docker 容器实例

此命令将创建一个配置好 Gaudi 环境的容器实例。

```bash
# 环境变量配置
NAME="video-gaudi-service"
IMG_NAME="text2video-gaudi:latest"
HTTP_PROXY="http://your-proxy-address:port"
HTTPS_PROXY="http://your-proxy-address:port"
HF_ENDPOINT="https://hf-mirror.com" # Hugging Face 模型下载镜像地址

# Gaudi 相关运行参数
RUN_ARG="-e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add SYS_PTRACE --cap-add=sys_nice --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --ipc=host --net=host --device=/dev:/dev -v /dev:/dev -v /sys/kernel/debug:/sys/kernel/debug"

# 创建并启动容器
echo "正在创建 Docker 实例: ${NAME}"
docker run -it --name ${NAME} \
  -p 9389:9389 \
  -v /mnt/disk2/HF_models:/hf \
  -e http_proxy=$HTTP_PROXY \
  -e https_proxy=$HTTPS_PROXY \
  -e HF_ENDPOINT=$HF_ENDPOINT \
  ${RUN_ARG} \
  --user root \
  --workdir=/home/user/text2video \
  ${IMG_NAME} /bin/bash
```

## 服务使用

### 1. 启动 Web 服务

在容器内部执行以下命令，启动 API 服务。

```bash
# 进入容器
docker exec -it video-gaudi-service bash

# 切换到工作目录并启动服务
cd /home/user/text2video
python3 web_service.py > web.log 2>&1 &
```

### 2. 启动 Gaudi 作业服务

此服务负责处理视频生成任务。

```bash
PT_HPU_SYNC_LAUNCH=1 PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node=8 --master-port 29502 --standalone job_service.py \
    --size infinitetalk-480 \
    --mode streaming \
    --motion_frame 9 \
    --offload_model False \
    --ulysses_size=8 > job.log 2>&1 &
```

---

## API 端点

> **内部网络使用说明:** 在公司内部网络调用此 API 时，请确保已正确设置 `no_proxy` 环境变量，以避免代理问题：
>
> ```bash
> export no_proxy="localhost,10.239.15.41,127.0.0.1,::1"
> ```

### 1. 创建视频

此端点基于文本提示、参考图像/视频和音频文件的组合来生成一个新视频。

- **端点:** `POST /v1/videos`
- **内容类型:** `multipart/form-data`

#### 请求参数

| 参数                | 类型          |  必需  | 默认值  | 描述                                                                  |
| ------------------- | ------------- | :----: | ------- | --------------------------------------------------------------------- |
| `input_reference`   | 文件          | **是** | N/A     | 源参考图像或视频文件。                                                |
| `audio` / `audio[]` | 文件 / [文件] | **是** | `[]`    | 用于生成的单个或多个音频文件。                                        |
| `prompt`            | 字符串        |   否   | `None`  | 用于指导视频生成的描述性文本提示。                                    |
| `audio_guide_scale` | 浮点数        |   否   | `5.0`   | 控制音频对生成过程的影响程度。                                        |
| `audio_type`        | 字符串        |   否   | `"add"` | 定义多个音频文件的处理方式。有效选项：`add` (叠加) 或 `para` (并行)。 |
| `fps`               | 整数          |   否   | `24`    | 生成视频的帧率（每秒帧数）。                                          |
| `shift`             | 浮点数        |   否   | `5.0`   | 一个特定的生成参数，用于控制视频动态。                                |
| `steps`             | 整数          |   否   | `50`    | 推理步数。                                                            |
| `seed`              | 整数          |   否   | `42`    | 用于可复现结果的随机种子。                                            |
| `guide_scale`       | 浮点数        |   否   | `5.0`   | 控制生成视频与提示的贴合程度。                                        |
| `logo_video`        | 布尔值        |   否   | `False` | 如果为 `True`，将自动附加一个 Intel 标志视频。                        |
| `seconds`           | 整数          |   否   | `20`    | **(注意)** 期望的视频长度（秒）。目前，实际长度由音频输入决定。       |

#### 响应体

成功的请求会将作业加入队列，并返回一个具有以下结构的 JSON 对象：

| 参数             | 类型   | 描述                                                                       |
| ---------------- | ------ | -------------------------------------------------------------------------- |
| `id`             | 字符串 | 视频生成作业的唯一标识符。                                                 |
| `object`         | 字符串 | 对象类型，始终为 `"video"`。                                               |
| `model`          | 字符串 | 用于生成的模型 (例如, `"InfiniteTalk"`)。                                  |
| `status`         | 字符串 | 作业的当前状态 (`queued`, `processing`, `completed`, `deleted`, `error`)。 |
| `progress`       | 整数   | 任务的大致完成百分比。                                                     |
| `created_at`     | 整数   | 作业创建时的 Unix 时间戳（秒）。                                           |
| `estimated_time` | 整数   | 预计完成时间（分钟）。                                                     |
| `queue_length`   | 整数   | 在此作业之前排队的作业数量。                                               |
| `duration`       | 整数   | 生成视频所花费的时间（秒）。                                               |
| `seconds`        | 整数   | 生成视频的最终时长（秒）。                                                 |
| `error`          | 字符串 | 解释失败原因的消息（如果有）。                                             |

<details>
<summary><strong>响应示例</strong></summary>

```json
{
  "id": "video_1766454718_2556",
  "object": "video",
  "model": "InfinteTalk",
  "status": "queued",
  "progress": 0,
  "created_at": 1766454718,
  "estimated_time": 14,
  "queue_length": 1,
  "duration": 0,
  "seconds": "3",
  "error": ""
}
```

</details>

---

### 2. 获取视频状态

检索视频生成作业的当前状态和进度。

- **端点:** `GET /v1/videos/{video_id}`

#### 响应体

返回与创建端点相同的 JSON 对象，但 `status` 和 `progress` 字段会更新。当 `status` 为 `"completed"` 时，表示视频已准备就绪。

<details>
<summary><strong>完成状态响应示例</strong></summary>

```json
{
  "id": "video_1766454718_2556",
  "object": "video",
  "model": "InfinteTalk",
  "status": "completed",
  "progress": 100,
  "created_at": 1766454718,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 430,
  "seconds": "3",
  "error": ""
}
```

</details>

---

### 3. 获取视频内容

下载生成的视频文件。

- **端点:** `GET /v1/videos/{video_id}/content`

此端点返回原始视频数据 (MIME 类型 `video/mp4`)，可以直接保存到文件中。

---

### 4. 删除视频

从服务器删除视频生成作业及其关联文件。

- **端点:** `DELETE /v1/videos/{video_id}`

> **注意:** 状态为 `processing` (处理中) 的作业无法被删除。

#### 响应体

成功删除后，服务器会返回一个包含作业最终元数据和 `"deleted"` 状态的 JSON 对象。

<details>
<summary><strong>删除状态响应示例</strong></summary>

```json
{
  "id": "video_1721105333_1234",
  "model": "InfinteTalk",
  "status": "deleted",
  "progress": 0,
  "created_at": 1721105333,
  "seconds": "15",
  "duration": 14,
  "estimated_time": 0,
  "queue_length": 0,
  "error": ""
}
```

</details>

---

## API 使用示例

### 示例 1: 提示 + 音频 + 图像

```bash
curl -X POST "http://10.239.15.41:9389/v1/videos" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=一个女人在录音棚里对着专业麦克风热情地唱歌..." \
  -F "input_reference=@examples/single/ref_image.png" \
  -F "audio=@examples/single/1.wav"
```

### 示例 2: 提示 + 音频 + 视频

```bash
curl -X POST "http://10.239.15.41:9389/v1/videos" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=一个男人在说话" \
  -F "input_reference=@examples/single/ref_video.mp4" \
  -F "audio=@examples/single/1.wav"
```

### 示例 3: 提示 + 多个音频 + 图像 (并行模式)

```bash
curl -X POST "http://10.239.15.41:9389/v1/videos" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=在一个轻松、亲密的环境中，一个男人和一个女人正在进行一场真诚的对话..." \
  -F "input_reference=@examples/multi/ref_img.png" \
  -F "audio_type=para" \
  -F "audio[]=@examples/multi/1-man.WAV" \
  -F "audio[]=@examples/multi/1-woman.WAV"
```

### 示例 4: 检查状态并下载

```bash
# 1. 使用创建请求返回的 ID 检查作业状态
curl http://10.239.15.41:9389/v1/videos/video_1765526104_4523

# 2. 当状态变为 "completed" 后，下载视频
curl http://10.239.15.41:9389/v1/videos/video_1765526104_4523/content -o video.mp4
```

### 示例 5: 删除视频

```bash
curl -X DELETE http://10.239.15.41:9389/v1/videos/video_1765526104_4523
```

---

## 错误处理

API 返回标准的 HTTP 状态码和一致的 JSON 错误体，以帮助诊断问题。

### 通用错误格式

```json
{
  "error": {
    "message": "错误的详细描述。",
    "code": "HTTP 状态码字符串 (例如, '400')。"
  }
}
```

### 常见错误

| 状态码 | 错误类型                   | 常见触发原因                         |
| :----- | -------------------------- | ------------------------------------ |
| `400`  | **错误请求 (Bad Request)** | 缺少必需参数、参数值无效或文件损坏。 |
| `404`  | **未找到 (Not Found)**     | 请求的 `video_id` 不存在。           |
| `500`  | **内部服务器错误**         | 处理过程中发生意外的服务器端故障。   |

<details>
<summary><strong>查看错误响应示例</strong></summary>

**400 错误请求示例:**

```json
{
  "error": {
    "message": "无效的参数类型：'seconds' 参数必须大于 0。",
    "code": "400"
  }
}
```

**404 未找到示例:**

```json
{
  "error": {
    "message": "ID 为 video_1721105333_1234 的视频未找到。",
    "code": "404"
  }
}
```

**500 内部服务器错误示例:**

```json
{
  "error": {
    "message": "内部服务器错误：组件加载器未初始化。",
    "code": "500"
  }
}
```

</details>

## 相关链接

- [OPEA 项目](https://github.com/opea-project/GenAIComps)
- [项目文档](https://opea-project.github.io/)
