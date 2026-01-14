# Wan Animate 服务

OPEA Wan Animate (角色动画) 微服务，用于根据参考图像和驱动视频生成角色动画视频。

## 概述

本项目提供 OPEA Wan Animate 组件的独立部署方案。它通过 REST API 提供先进的角色动画生成能力，支持将参考图像中的角色按照驱动视频的动作进行动画化，或替换驱动视频中的角色。本服务针对英特尔 ® Habana® Gaudi® 加速器进行了优化。

## 主要特性

- **角色动画 (Animate)**: 将参考图像中的角色按照驱动视频的动作进行动画化。
- **角色替换 (Replace)**: 用参考图像中的角色替换驱动视频中的人物。
- **姿态重定向**: 自动将驱动视频的姿态映射到参考图像角色。
- **多分辨率支持**: 支持 720P 和 480P 两种分辨率输出。
- **任务队列管理**: 高效管理并发请求，确保服务稳定性。
- **HPU/Gaudi 优化**: 充分利用 Habana Gaudi 加速器的高性能计算能力。
- **RESTful API**: 提供标准化 RESTful 接口。
- **容器化部署**: 支持 Docker 快速部署和环境隔离。

## 安装部署

### 0. 硬件资源配置

建议使用至少 4 卡部署，性能模式推荐 8 卡。Animate-14B 模型较大，需要足够的显存。

### 1. 构建 Docker 镜像

在构建镜像前，请根据您的网络环境设置代理（如果需要）。

```bash
# 设置代理（可选）
export http_proxy="http://your-proxy-address:port"
export https_proxy="http://your-proxy-address:port"

# 克隆 optimum habana fork aice v1.22.0 分支
git clone https://github.com/HabanaAI/optimum-habana-fork.git -b aice/v1.22.0
# 进入 InfiniteTalk 目录
cd optimum-habana-fork/examples/InfiniteTalk/opea_text2video/

# 执行构建命令
docker build -t animate-gaudi:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f Dockerfile-animate .
```

### 2. 下载模型

请下载如下模型到指定的模型目录（通过环境变量 `HF_MODEL_PATH` 指定）：

```bash
export HF_MODEL_PATH=your_model_path

# 下载 Wan2.2-Animate-14B 主模型
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ${HF_MODEL_PATH}/Wan2.2-Animate-14B

# 预处理模型已包含在主模型目录的 process_checkpoint 子目录中
```

### 3. Docker Compose 部署（推荐）

#### 3.1 安装 Docker Compose v2

```bash
# Ubuntu 22.04 安装命令
sudo apt update
sudo apt install docker-compose-v2
```

#### 3.2 配置环境变量

参考 env_example 创建 `.env` 文件并配置以下变量：

```bash
# 模型路径（必需）
HF_MODEL_PATH=/mnt/disk2/HF_models

# 视频输出目录（可选）
VIDEO_OUTPUT_DIR=./video_output

# Web 服务端口（默认 9397）
WEB_PORT=9397

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
| `WEB_PORT`         | Web API 服务端口          | `9397`                    |
| `NUM_CARDS`        | 使用的 HPU 卡数，4 或者 8 | `4`                       |
| `https_proxy`      | HTTPS 代理地址            | -                         |
| `no_proxy`         | 不使用代理的地址列表      | `localhost,127.0.0.1,::1` |

#### 3.3 启动服务

```bash
# 启动服务（后台运行）
docker compose -f docker-compose-animate.yml up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f animate-web
tail -f ${HF_MODEL_PATH}/logs/animate_job.log
tail -f ${HF_MODEL_PATH}/logs/animate_web.log
```

服务启动后需等待 **5-8 分钟**模型加载完成（包括预处理模型和生成模型），可通过日志验证：

```bash
# 查看实时日志出现如下信息表示可以提供服务
tail -f ${HF_MODEL_PATH}/logs/animate_job.log
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9397 (Press CTRL+C to quit)
[2026-01-14 10:00:00,000] [    INFO] - animate - Animate service started.
```

#### 3.4 停止服务

```bash
# 停止服务
docker compose down
```

### 4. 手工方式

#### 4.1 创建 Docker 容器实例

```bash
# 环境变量配置
NAME="animate-gaudi-service"
IMG_NAME="animate-gaudi:latest"
HTTP_PROXY="http://your-proxy-address:port"
HTTPS_PROXY="http://your-proxy-address:port"
HF_ENDPOINT="https://hf-mirror.com"

# Gaudi 相关运行参数
RUN_ARG="-e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add SYS_PTRACE --cap-add=sys_nice --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --ipc=host --net=host --device=/dev:/dev -v /dev:/dev -v /sys/kernel/debug:/sys/kernel/debug"

# 创建并启动容器
echo "正在创建 Docker 实例: ${NAME}"
docker run -it --name ${NAME} \
  -p 9397:9397 \
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

```bash
# 进入容器
docker exec -it animate-gaudi-service bash

# 切换到工作目录并启动服务
cd /home/user/text2video
python3 web_service_animate.py --model_name_or_path Wan2.2-Animate-14B --rank_size ${HPU} 2>&1 &
```

#### 4.3 启动 Gaudi 作业服务

```bash
PT_HPU_RECIPE_CACHE_CONFIG=/home/user/cache_animate,false,40960 \
PT_HPU_SYNC_LAUNCH=1 \
PT_HPU_GPU_MIGRATION=1 \
PT_HPU_LAZY_MODE=1 \
torchrun --nproc_per_node=4 --standalone \
  job_service_animate.py \
  --ckpt_dir /hf/Wan2.2-Animate-14B \
  --process_ckpt_dir /hf/Wan2.2-Animate-14B/process_checkpoint \
  --ulysses_size 4 \
  --sample_solver dpm++ \
  > animate_job.log 2>&1 &
```

---

## API 端点

> **内部网络使用说明:** 在公司内部网络调用此 API 时，请确保已正确设置 `no_proxy` 环境变量：
>
> ```bash
> export no_proxy="localhost,10.239.15.47,127.0.0.1,::1"
> ```

### 1. 创建动画视频

此端点基于参考图像和驱动视频生成角色动画视频。

- **端点:** `POST /v1/animate`
- **内容类型:** `multipart/form-data`

#### 请求参数

| 参数         | 类型   | 必需 | 默认值               | 可选值                                       | 描述                                                           |
| ------------ | ------ | :--: | -------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| `image`      | 文件   |  是  | -                    | -                                            | 参考图像文件，包含要动画化的角色。                             |
| `video`      | 文件   |  是  | -                    | -                                            | 驱动视频文件，提供动作来源。                                   |
| `prompt`     | 字符串 |  否  | `视频中的人在做动作` | -                                            | 用于指导视频生成的描述性文本提示。                             |
| `mode`       | 字符串 |  否  | `animate`            | `animate`, `replace`                         | 动画模式。`animate`: 动作迁移；`replace`: 角色替换。           |
| `size`       | 字符串 |  否  | `1280*720`           | `1280*720`, `720*1280`, `832*480`, `480*832` | 输出视频分辨率。                                               |
| `seconds`    | 整数   |  否  | `2`                  | -                                            | 预期视频时长（秒），用于时间估算。实际输出长度由驱动视频决定。 |
| `refert_num` | 整数   |  否  | `1`                  | `1`, `5`                                     | 时序引导帧数。1=更快；5=更好的时序一致性。                     |
| `seed`       | 整数   |  否  | `-1`                 | -                                            | 随机种子。-1 表示随机生成。                                    |
| `shift`      | 浮点数 |  否  | `5.0`                | -                                            | 噪声调度偏移参数。                                             |
| `steps`      | 整数   |  否  | `20`                 | -                                            | 扩散采样步数。数值越高质量越好，但速度越慢。                   |

**模式说明：**

- **animate (动画模式)**: 将驱动视频的动作迁移到参考图像中的角色，生成该角色执行相同动作的视频。
- **replace (替换模式)**: 用参考图像中的角色替换驱动视频中的人物，保持原视频的背景和场景。

**分辨率说明：**

| 尺寸字符串 | 宽度 | 高度 | 分辨率 | 纵横比      |
| ---------- | ---- | ---- | ------ | ----------- |
| `1280*720` | 1280 | 720  | 720P   | 16:9 (横屏) |
| `720*1280` | 720  | 1280 | 720P   | 9:16 (竖屏) |
| `832*480`  | 832  | 480  | 480P   | 16:9 (横屏) |
| `480*832`  | 480  | 832  | 480P   | 9:16 (竖屏) |

> **注意:** 输出视频的实际时长由驱动视频的长度决定，而不是 `seconds` 参数。`seconds` 参数仅用于队列时间估算。建议使用 2-5 秒的短视频作为驱动源以获得最佳效果。

#### 响应体

成功的请求会将作业加入队列，并返回一个具有以下结构的 JSON 对象：

| 参数             | 类型   | 描述                                                                       |
| ---------------- | ------ | -------------------------------------------------------------------------- |
| `id`             | 字符串 | 视频生成作业的唯一标识符。                                                 |
| `object`         | 字符串 | 对象类型，始终为 `"video"`。                                               |
| `model`          | 字符串 | 用于生成的模型 (例如, `"Wan2.2-Animate-14B"`)。                            |
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
  "model": "Wan2.2-Animate-14B",
  "status": "queued",
  "progress": 0,
  "created_at": 1767068943,
  "estimated_time": 5,
  "queue_length": 1,
  "duration": 0,
  "seconds": "2",
  "error": ""
}
```

</details>

---

### 2. 获取视频状态

检索视频生成作业的当前状态和进度。

- **端点:** `GET /v1/animate/{video_id}`

#### 响应体

返回与创建端点相同的 JSON 对象，但 `status` 和 `progress` 字段会更新。当 `status` 为 `"completed"` 时，表示视频已准备就绪。

<details>
<summary><strong>处理中状态响应示例</strong></summary>

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "processing",
  "progress": 45,
  "created_at": 1767068943,
  "estimated_time": 3,
  "queue_length": 0,
  "duration": 0,
  "seconds": "2",
  "error": ""
}
```

</details>

<details>
<summary><strong>完成状态响应示例</strong></summary>

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "completed",
  "progress": 100,
  "created_at": 1767068943,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 120,
  "seconds": "2",
  "error": ""
}
```

</details>

---

### 3. 获取视频内容

下载生成的视频文件。

- **端点:** `GET /v1/animate/{video_id}/content`

此端点返回原始视频数据 (MIME 类型 `video/mp4`)，可以直接保存到文件中。

---

### 4. 删除视频

从服务器删除视频生成作业及其关联文件。

- **端点:** `DELETE /v1/animate/{video_id}`

> **注意:** 状态为 `processing` (处理中) 的作业无法被删除。

#### 响应体

成功删除后，服务器会返回一个包含作业最终元数据和 `"deleted"` 状态的 JSON 对象。

<details>
<summary><strong>删除状态响应示例</strong></summary>

```json
{
  "id": "video_1767068943_3357",
  "object": "video",
  "model": "Wan2.2-Animate-14B",
  "status": "deleted",
  "progress": 0,
  "created_at": 1767068943,
  "estimated_time": 0,
  "queue_length": 0,
  "duration": 0,
  "seconds": "2",
  "error": ""
}
```

</details>

---

## API 使用示例

### 示例 1: 动画模式 - 将参考图像角色动画化

```bash
curl -X POST "http://localhost:9397/v1/animate" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@reference_person.jpg" \
  -F "video=@driving_motion.mp4" \
  -F "mode=animate" \
  -F "seconds=3"
```

### 示例 2: 替换模式 - 替换视频中的角色

```bash
curl -X POST "http://localhost:9397/v1/animate" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@new_character.jpg" \
  -F "video=@original_video.mp4" \
  -F "mode=replace" \
  -F "size=1280*720"
```

### 示例 3: 高质量设置

```bash
curl -X POST "http://localhost:9397/v1/animate" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@character.jpg" \
  -F "video=@dance.mp4" \
  -F "mode=animate" \
  -F "refert_num=5" \
  -F "steps=30" \
  -F "seed=42"
```

### 示例 4: 竖屏视频输出

```bash
curl -X POST "http://localhost:9397/v1/animate" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@portrait.jpg" \
  -F "video=@vertical_motion.mp4" \
  -F "size=720*1280"
```

### 示例 5: 检查状态并下载

```bash
# 1. 使用创建请求返回的 ID 检查作业状态
curl http://localhost:9397/v1/animate/video_1767068943_3357

# 2. 当状态变为 "completed" 后，下载视频
curl http://localhost:9397/v1/animate/video_1767068943_3357/content -o animated_video.mp4
```

### 示例 6: 删除视频

```bash
curl -X DELETE http://localhost:9397/v1/animate/video_1767068943_3357
```

---

## 处理流程

Animate 服务包含两个主要阶段：

### 1. 预处理阶段

- **姿态提取**: 从驱动视频中提取每帧的人体姿态关键点。
- **人脸区域提取**: 提取驱动视频中的人脸区域用于表情引导。
- **参考图像处理**: 对参考图像进行姿态检测和尺寸调整。
- **姿态重定向** (animate 模式): 将驱动视频的姿态映射到参考图像角色的身体比例。
- **遮罩生成** (replace 模式): 生成用于角色替换的分割遮罩和背景。

### 2. 生成阶段

- **扩散模型推理**: 使用预处理数据生成最终视频帧。
- **时序引导**: 利用 `refert_num` 参数控制帧间一致性。
- **视频合成**: 将生成的帧合成为最终视频文件。

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

| 状态码 | 错误类型                   | 常见触发原因                                           |
| :----- | -------------------------- | ------------------------------------------------------ |
| `400`  | **错误请求 (Bad Request)** | 缺少必需参数、参数值无效、文件格式不支持或文件损坏。   |
| `404`  | **未找到 (Not Found)**     | 请求的 `video_id` 不存在。                             |
| `500`  | **内部服务器错误**         | 预处理失败（如未检测到人脸）或生成过程中发生意外错误。 |

<details>
<summary><strong>查看错误响应示例</strong></summary>

**400 错误请求示例 - 缺少参数:**

```json
{
  "error": {
    "message": "Missing required parameter: image",
    "code": "400"
  }
}
```

**400 错误请求示例 - 无效参数:**

```json
{
  "error": {
    "message": "Invalid refert_num: 3. Must be 1 or 5.",
    "code": "400"
  }
}
```

**404 未找到示例:**

```json
{
  "error": {
    "message": "Video with id video_1767068943_3357 not found.",
    "code": "404"
  }
}
```

**500 内部服务器错误示例:**

```json
{
  "error": {
    "message": "Preprocessing failed: No face detected in reference image",
    "code": "500"
  }
}
```

</details>

---

## 最佳实践

### 输入图像建议

- 使用清晰、高分辨率的参考图像。
- 确保参考图像中的人物面部清晰可见。
- 对于 animate 模式，建议人物姿态与驱动视频首帧相似。
- 避免使用遮挡严重或光照极端的图像。

### 驱动视频建议

- 使用稳定、清晰的视频作为驱动源。
- 确保驱动视频中只有一个主要人物。
- 避免快速运动或频繁切换的视频。
- 视频时长建议在 2-5 秒之间以获得最佳效果。

### 参数调优

- **快速预览**: `refert_num=1`, `steps=15`
- **平衡模式**: `refert_num=1`, `steps=20` (默认)
- **高质量模式**: `refert_num=5`, `steps=30`

---

## 服务启动配置

以下参数在服务启动时设置，**用户不可配置**：

| 参数                 | 说明                                | 默认值                                      |
| -------------------- | ----------------------------------- | ------------------------------------------- |
| `--ckpt_dir`         | Wan2.2-Animate-14B 模型检查点路径   | `/hf/Wan2.2-Animate-14B`                    |
| `--process_ckpt_dir` | 预处理模型检查点路径                | `/hf/Wan2.2-Animate-14B/process_checkpoint` |
| `--ulysses_size`     | 多卡推理的序列并行大小              | `1`                                         |
| `--video_dir`        | 生成视频的输出目录                  | `/home/user/video`                          |
| `--sample_solver`    | 采样求解器算法 (`unipc` 或 `dpm++`) | `dpm++`                                     |
| `--sep`              | 作业文件字段分隔符                  | `,`                                         |

---

## 文件结构

```
opea_text2video/
├── src/
│   ├── web_service_animate.py    # FastAPI Web 服务
│   ├── job_service_animate.py    # 后台作业处理服务
│   ├── component_animate.py      # OPEA 组件定义
│   └── util.py                   # 共享工具函数
├── Dockerfile-animate            # Docker 构建文件
├── docker-compose-animate.yml    # Docker Compose 配置
├── README_animate.md             # 本文档
└── design_animate.md             # 设计文档
```
