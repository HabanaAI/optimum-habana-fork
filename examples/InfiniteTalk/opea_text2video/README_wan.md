# Text2Video 服务

OPEA Text-to-Video (文本到视频) 微服务，用于根据文本提示和音频输入生成视频。

## 概述

本项目提供 OPEA Text2Video 组件的独立部署方案。它通过 REST API 提供先进的视频生成能力，并针对英特尔 ® Habana® Gaudi® 加速器进行了优化。本指南提供了基于 docker compose 的自动部署和命令行的手工部署两种部署方式。

## 主要特性

- **文生视频**: 支持文本提示和音频条件输入，生成动态视频。
- **任务队列管理**: 高效管理并发请求，确保服务稳定性。
- **HPU/Gaudi 优化**: 充分利用 Habana Gaudi 加速器的高性能计算能力。
- **RESTful API**: 提供标准化 RESTful 接口及 OpenAPI 类似接口。
- **容器化部署**: 支持 Docker 快速部署和环境隔离。

## 安装部署

### 0. 硬件资源配置

建议使用至少 4 卡部署，性能模式推荐 8 卡。对于 Gaudi2E 8 卡互联，请使用专用优化镜像部署该服务。

### 1. 构建 Docker 镜像

在构建镜像前，请根据您的网络环境设置代理（如果需要）。

```bash
# 设置代理（可选）
export http_proxy="http://your-proxy-address:port"
export https_proxy="http://your-proxy-address:port"

# 克隆 optimum habana fork aice v1.22.0 分支
git clone https://github.com/HabanaAI/optimum-habana-fork.git -b aice/v1.22.0
# 进入 InfiniteTalk目录
cd optimum-habana-fork/examples/InfiniteTalk/opea_text2video/

# 执行构建命令
docker build -t text2video-gaudi:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f Dockerfile-wan .
```

### 2. 下载模型

请下载如下模型到指定的模型目录（通过环境变量 `HF_MODEL_PATH` 指定）：

```bash
export HF_MODEL_PATH=your_model_path
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ${HF_MODEL_PATH}/Wan2.2-TI2V-5B
```

### 3. Docker Compose 部署（推荐）

#### 3.1 安装 Docker Compose v2

```bash
# Ubuntu 22.04 安装命令
sudo apt update
sudo apt install docker-compose-v2
```

如遇到 docker 版本过低问题，请查询 DeepSeek ubuntu 22.04 升级 docker 和 docker-compose-v2 进行操作。

#### 3.2 配置环境变量

参考 env_example 创建 `.env` 文件并配置以下变量：

```bash
# 模型路径（必需）
HF_MODEL_PATH=/mnt/disk2/HF_models

# 视频输出目录（可选）
VIDEO_OUTPUT_DIR=./video_output

# Web 服务端口（默认 9396）
WEB_PORT=9396

# HPU 卡数（默认 4）
NUM_CARDS=4

# 代理设置（可选）
https_proxy=http://your-proxy:port
no_proxy=localhost,127.0.0.1,::1
```

**环境变量说明：**

| 变量               | 说明                      | 默认值                    |
| ------------------ | ------------------------- | ------------------------- |
| `HF_MODEL_PATH`    | HuggingFace 模型存储路径  | -                         |
| `VIDEO_OUTPUT_DIR` | 生成视频的输出目录        | `./video_output`          |
| `WEB_PORT`         | Web API 服务端口          | `9396`                    |
| `NUM_CARDS`        | 使用的 HPU 卡数，4 或者 8 | `4`                       |
| `https_proxy`      | HTTPS 代理地址            | -                         |
| `no_proxy`         | 不使用代理的地址列表      | `localhost,127.0.0.1,::1` |

#### 3.3 启动服务

```bash
# 启动服务（后台运行）
docker compose -f docker-compose-wan.yml up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f text2video-web
tail -f  ${HF_MODEL_PATH}/logs/job.log
tail -f  ${HF_MODEL_PATH}/logs/web.log
```

服务启动后需等待 **3-5 分钟**模型加载完成，可通过健康检查命令验证：

```bash

# 查看实时日志出现如下信息表示可以提供服务
tail -f  ${HF_MODEL_PATH}/logs/job.log
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9396 (Press CTRL+C to quit)
[2025-12-25 17:02:10,081] [    INFO] - Base service - HTTP server setup successful
[2025-12-25 17:02:10,084] [    INFO] - text2video - Text-to-video server started.

```

#### 3.4 停止服务

```bash
# 停止服务
docker compose down

```

### 4. 手工方式

#### 4.1 创建 Docker 容器实例

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

#### 4.2 启动 Web 服务

在容器内部执行以下命令，启动 API 服务。

```bash
# 进入容器
docker exec -it video-gaudi-service bash

# 切换到工作目录并启动服务
cd /home/user/text2video
# HPU 设置为使用的卡数
python3 web_service.py --model_name_or_path Wan2.2-TI2V-5B --rank_size ${HPU} 2>&1 &
```

#### 4.3 启动 Gaudi 作业服务

此服务负责处理视频生成任务。

下面的示例使用 4 卡来启动 Gaudi 作业服务。

```bash
PT_HPU_RECIPE_CACHE_CONFIG=/home/user/cache_wan,false,40960 PT_HPU_SYNC_LAUNCH=1 PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun --nproc_per_node=4 --standalone job_service_wan.py --ulysses_size 4 > job.log 2>&1 &
```

---

## API 端点

> **内部网络使用说明:** 在公司内部网络调用此 API 时，请确保已正确设置 `no_proxy` 环境变量，以避免代理问题：
>
> ```bash
> export no_proxy="localhost,10.239.15.47,127.0.0.1,::1"
> ```

目前部署了一个实例： G11: IP addr: 10.239.15.47 Wan2.2-TI2V-5B text/image to video service

### 1. 创建视频

此端点基于文本提示、参考图像/视频和音频文件的组合来生成一个新视频。

- **端点:** `POST /v1/videos`
- **内容类型:** `multipart/form-data`

#### Wan2.2-TI2V-5B text/image to video 请求参数

| 参数              | 类型   | 必需 | 默认值  | 描述                                                                                          |
| ----------------- | ------ | :--: | ------- | --------------------------------------------------------------------------------------------- |
| `input_reference` | 文件   |  否  | N/A     | 源参考图像文件。                                                                              |
| `prompt`          | 字符串 |  否  | ``      | 用于指导视频生成的描述性文本提示。                                                            |
| `steps`           | 整数   |  否  | `50`    | 推理步数。                                                                                    |
| `seed`            | 整数   |  否  | `42`    | 用于可复现结果的随机种子。                                                                    |
| `portrait`        | 布尔值 |  否  | `False` | 是否为竖屏显示，默认生成的视频是横屏（1280\*704），此参数只适用于文生视频，图生视频此参数无效 |
| `guide_scale`     | 浮点数 |  否  | `5.0`   | 控制生成视频与提示的贴合程度。                                                                |
| `seconds`         | 整数   |  否  | `5`     | 期望的视频长度（秒）。                                                                        |

注意：

1. input_reference 和 prompt 不能同时为空
2. 对于图生视频，将对图片大小自动 scale 到 720p 范围

#### 响应体

成功的请求会将作业加入队列，并返回一个具有以下结构的 JSON 对象：

| 参数             | 类型   | 描述                                                                       |
| ---------------- | ------ | -------------------------------------------------------------------------- |
| `id`             | 字符串 | 视频生成作业的唯一标识符。                                                 |
| `object`         | 字符串 | 对象类型，始终为 `"video"`。                                               |
| `model`          | 字符串 | 用于生成的模型 (例如, `"Wan2.2-TI2V-5B"`)。                                |
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
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-TI2V-5B",
  "status": "queued",
  "progress": 0,
  "created_at": 1767068943,
  "estimated_time": 7,
  "queue_length": 1,
  "duration": 0,
  "seconds": "8",
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
  "id": "video_1767511247_1919",
  "object": "video",
  "model": "Wan2.2-TI2V-5B",
  "status": "completed",
  "progress": 100,
  "created_at": 1767511247,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 105,
  "seconds": "5",
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
  "id": "video_1767511247_1919",
  "object": "video",
  "model": "Wan2.2-TI2V-5B",
  "status": "deleted",
  "progress": 0,
  "created_at": 1767511247,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 105,
  "seconds": "5",
  "error": ""
}
```

</details>

---

## API 使用示例

### 示例 1: 文本 -> 生成视频

```bash
curl -X POST "http://10.239.15.47:9396/v1/videos" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=一个女人在录音棚里对着专业麦克风热情地唱歌..."
```

### 示例 2: 图片 -> 生成视频

```bash
curl -X POST "http://10.239.15.47:9396/v1/videos" \
  -H "Content-Type: multipart/form-data" \
  -F "input_reference=@examples/i2v_input.JPG"
```

### 示例 3: 文本 + 图片 -> 生成视频

```bash
curl -X POST "http://10.239.15.47:9396/v1/videos" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=一个男人在说话" \
  -F "input_reference=@examples/i2v_input.JPG"
```

### 示例 4: 检查状态并下载

```bash
# 1. 使用创建请求返回的 ID 检查作业状态
curl http://10.239.15.47:9396/v1/videos/video_1765526104_4523

# 2. 当状态变为 "completed" 后，下载视频
curl http://10.239.15.47:9396/v1/videos/video_1765526104_4523/content -o video.mp4
```

### 示例 5: 删除视频

```bash
curl -X DELETE http://10.239.15.47:9396/v1/videos/video_1765526104_4523
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
