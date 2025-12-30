# 图像生成与编辑服务

OPEA 图像生成与编辑微服务，提供基于文本提示生成图像和图像编辑的能力。

## 概述

本项目提供三个独立的图像处理容器化 AI 服务，均针对英特尔® Habana® Gaudi® 加速器进行了优化：

1. **图像编辑服务（Qwen-Image-Edit-2509）**：基于 Qwen-Image-Edit-2509 模型支持文本描述对输入图片进行编辑。
2. **图像生成服务（Qwen-Image）**：基于 Qwen-Image 模型的文本生成图像服务。
3. **图像生成服务（Z-Image-Turbo）**：基于 Z-Image-Turbo 模型的快速高吞吐文本生成图像服务。

## 主要特性

- **文本生成图像**：支持根据文本描述生成高质量图像
- **图像编辑**：支持输入图片+文本提示进行图像编辑
- **HPU/Gaudi 优化**：充分利用 Habana Gaudi 加速器的高性能计算能力
- **RESTful API**：提供标准化 RESTful 接口，兼容 OpenAI API 格式
- **容器化部署**：支持 Docker 快速部署和环境隔离
- **灵活质量控制**：支持 high/medium/low 三种质量级别，自动调整推理步数

## 支持的模型及功能

| 模型名称 | 服务类型 | 功能描述 | API 端点 |
|---------|---------|---------|----------|
| **Qwen-Image-Edit-2509** | 图像编辑 | 基于输入图片和文本提示进行图像编辑，支持添加、修改图像内容 | `POST /v1/images/edits` |
| **Qwen-Image** | 图像生成 | 文本到图像生成，高质量输出，适合创意设计场景 | `POST /v1/images/generations` |
| **Z-Image-Turbo** | 图像生成 | 文本到图像生成，针对速度优化，适合高吞吐量场景 | `POST /v1/images/generations` |

### 模型功能对比

- **Qwen-Image-Edit-2509**：适用于图像修改、内容添加、风格转换等场景，需要原始图片作为输入
- **Qwen-Image**：适用于高质量图像生成，提供更精细的图像质量
- **Z-Image-Turbo**：适用于快速生成场景，推理速度更快，步数更少（默认9步）

## 硬件资源配置

- **推荐配置**：每个模型1张 Intel Gaudi/HPU 卡

## 安装部署

### 0. 部署前准备

**系统要求：**
- Docker 和 Docker Compose v2 已安装
- Intel Gaudi 驱动和运行时已安装
- 具备 HPU 硬件（Gaudi/Gaudi2）

**网络配置：**
```bash
# 如需使用代理，请配置
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
export no_proxy=localhost,127.0.0.1,::1
```

### 1. 下载模型

请下载所需模型到指定目录（默认：`/mnt/disk4/hf_models`）：

```bash
# 设置模型下载路径
export HF_MODEL_PATH=/mnt/disk4/hf_models

# 下载 Qwen-Image-Edit-2509 模型（图像编辑服务）
pip install modelscope
modelscope download Qwen/Qwen-Image-Edit-2509 --local_dir ${HF_MODEL_PATH}/Qwen-Image-Edit-2509

# 下载 Qwen-Image 模型（图像生成服务）
modelscope download Qwen/Qwen-Image --local_dir ${HF_MODEL_PATH}/Qwen-Image

# 下载 Z-Image-Turbo 模型（高性能图像生成服务）
modelscope download Tongyi-MAI/Z-Image-Turbo --local-dir ${HF_MODEL_PATH}/Z-Image-Turbo
```

### 2. 下载代码并配置环境变量

下载代码
```bash
# 设置代理（可选）

# 克隆 optimum habana fork aice v1.22.0 分支
git clone https://github.com/HabanaAI/optimum-habana-fork.git -b aice/v1.22.0
# 进入 InfiniteTalk目录
cd optimum-habana-fork/examples/stable-diffusion/opea_deployment/


参考 `env_example` 创建 `.env` 文件并配置以下变量：

```bash
# 模型路径（必需）
HOST_MODEL_DIR=/mnt/disk4/hf_models

# 代理设置（可选）
https_proxy=http://child-prc.intel.com:912
no_proxy=localhost,127.0.0.1,::1

# HPU 设备分配（可选，指定每个服务使用的 HPU 设备 ID）
HABANA_QWEN_IMAGE_DEVICE=0
HABANA_QWEN_IMAGE_EDIT_DEVICE=1
HABANA_ZIMAGE_EDIT_DEVICE=2

# 服务端口配置（可选）
PORT_ZIMG=9392
PORT_QWEN_IMG=9391
PORT_QWEN_EDIT=9390
```

#### 环境变量详细说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| **通用环境变量** |
| `HOST_MODEL_DIR` | 主机模型存储路径 | `/mnt/disk4/hf_models` |
| `https_proxy` | HTTPS 代理地址 | 空 |
| `no_proxy` | 不使用代理的地址列表 | `localhost,127.0.0.1,::1` |
| **HPU 设备分配** |
| `HABANA_QWEN_IMAGE_DEVICE` | Qwen-Image 服务使用的 HPU 设备 ID, 0-7 | `0` |
| `HABANA_QWEN_IMAGE_EDIT_DEVICE` | Qwen-Image-Edit 服务使用的 HPU 设备 ID, 0-7 | `1` |
| `HABANA_ZIMAGE_EDIT_DEVICE` | Z-Image-Turbo 服务使用的 HPU 设备 ID, 0-7 | `2` |
| **服务端口配置** |
| `PORT_ZIMG` | Z-Image-Turbo 服务端口 | `9392` |
| `PORT_QWEN_IMG` | Qwen-Image 服务端口 | `9391` |
| `PORT_QWEN_EDIT` | Qwen-Image-Edit 服务端口 | `9390` |

### 3. 构建 Docker 镜像

```bash
# 构建所有服务镜像
docker compose build

# 或构建单个服务镜像
docker compose build images-edits-qwen-image-edit-api-server
docker compose build images-generations-qwen-image-api-server
docker compose build images-generations-z-image-turbo-api-server
```

### 4. 启动服务

#### 启动所有服务

```bash
docker compose up -d
```

#### 启动单个服务

```bash
# 启动图像编辑服务（端口 9390）
docker compose up images-edits-qwen-image-edit-api-server -d

# 启动 Qwen-Image 图像生成服务（端口 9391）
docker compose up images-generations-qwen-image-api-server -d

# 启动 Z-Image-Turbo 图像生成服务（端口 9392）
docker compose up images-generations-z-image-turbo-api-server -d
```

### 5. 验证服务状态

#### 查看容器状态

```bash
docker ps
```

#### 查看服务日志

**图像编辑服务日志：**
```bash
docker compose logs images-edits-qwen-image-edit-api-server
```

等待出现如下日志表示服务就绪：
```
[    INFO] - images_edits - images_edits server started
```

**Qwen-Image 图像生成服务日志：**
```bash
docker compose logs images-generations-qwen-image-api-server
```

等待出现如下日志表示服务就绪：
```
[    INFO] - images_generations - images_generations server started
```

**Z-Image-Turbo 图像生成服务日志：**
```bash
docker compose logs images-generations-z-image-turbo-api-server
```

等待出现如下日志表示服务就绪：
```
[    INFO] - images_generations - images_generations server started
```

### 6. 停止服务

```bash
# 停止所有服务
docker compose down

# 停止单个服务
docker compose down images-edits-qwen-image-edit-api-server
docker compose down images-generations-qwen-image-api-server
docker compose down images-generations-z-image-turbo-api-server
```

## 各模型特别配置

这部分内容为微服务启动参数说明，无需做改动，自动配置,无特别需求不需要改变。
### 1. Qwen-Image-Edit-2509（图像编辑服务）

**启动命令：**
```bash
python3 opea_images_edits_microservice.py \
  --device hpu \
  --bf16 \
  --model_name_or_path /workspace/data/Qwen-Image-Edit-2509
```

**特殊环境变量：**
```bash
QWEN25VL_FP32_SOFTMAX=True  # 必需，用于 Qwen 模型正确计算
```

### 2. Qwen-Image（图像生成服务）

**启动命令：**
```bash
python3 opea_images_generations_microservice.py \
  --device hpu \
  --bf16 \
  --model_name_or_path /workspace/data/Qwen-Image
```

**特殊环境变量：**
```bash
QWEN25VL_FP32_SOFTMAX=True  # 必需，用于 Qwen 模型正确计算
```

**特性：**
- 高质量图像生成
- 默认 25 步推理（medium 质量）
- 支持自定义 seed、guidance scale 等参数

### 3. Z-Image-Turbo（高性能图像生成服务）

**启动命令：**
```bash
python3 opea_images_generations_microservice.py \
  --device hpu \
  --bf16 \
  --guidance_scale 0.0 \
  --num_inference_steps 9 \
  --model_name_or_path /workspace/data/Z-Image-Turbo
```

**特殊环境变量：**
```bash
PT_HPU_GPU_MIGRATION=1    # 必需，启用 GPU 迁移优化
USE_ZIMAGE_BUCKET=1       # 必需，使用 Z-Image 专用 bucket
```

**特性：**
- 支持 BF16 精度计算
- 针对速度优化，默认 9 步推理
- guidance_scale 设置为 0.0（无需分类器引导）

## API 接口文档

### 1. 图像生成 API（`/v1/images/generations`）

**适用模型：** Qwen-Image、Z-Image-Turbo

**端点：** `POST http://localhost:9391/v1/images/generations`（Qwen-Image）
或 `POST http://localhost:9392/v1/images/generations`（Z-Image-Turbo）

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | 是 | 描述希望生成的图片内容的文本提示 |
| `quality` | string | 否 | 图片质量：`high`（高）、`medium`（中）、`low`（低） |
| `size` | string | 否 | 图片尺寸，格式 "宽x高"，如 "1024x1024" |
| `n` | int | 否 | 生成图片数量，默认 1 |

#### quality 参数与推理步数对应关系

**Z-Image-Turbo 模型：**

| quality | num_inference_steps |
|---------|---------------------|
| high | 20 |
| medium | 9 |
| low | 5 |
| 未指定或其它 | 9 |

**Qwen-Image 和其他模型：**

| quality | num_inference_steps |
|---------|---------------------|
| high | 50 |
| medium | 25 |
| low | 10 |
| 未指定或其它 | 25 |

#### 请求示例

```bash
curl -X POST http://localhost:9391/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat sitting on a bench in the park",
    "quality": "high",
    "size": "1024x1536",
    "n": 2
  }'
```

---

### 2. 图像编辑 API（`/v1/images/edits`）

**适用模型：** Qwen-Image-Edit-2509

**端点：** `POST http://localhost:9390/v1/images/edits`

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 待编辑的原始图片（支持单张或多张） |
| `prompt` | string | 是 | 编辑图片的文本描述 |
| `quality` | string | 否 | 图片生成质量：`high`、`medium`、`low` |
| `size` | string | 否 | 输出图片尺寸，格式 "宽x高" |
| `n` | int | 否 | 每张输入图片生成的数量，默认 1 |

**传输格式：** `multipart/form-data`

#### quality 参数与推理步数对应关系

| quality | num_inference_steps |
|---------|---------------------|
| high | 40 |
| medium | 20 |
| low | 10 |
| 未指定或其它 | 20 |

#### 请求示例

```bash
curl -X POST http://localhost:9390/v1/images/edits \
  -F "image=@/path/to/your/image.jpg" \
  -F "prompt=add blue sky and white clouds" \
  -F "quality=high" \
  -F "size=1024x1536" \
  -F "n=2"
```

---

### 3. 模型列表 API（`/v1/models`）

**端点：**
- `GET http://localhost:9390/v1/models`（图像编辑服务）
- `GET http://localhost:9391/v1/models`（Qwen-Image 图像生成服务）
- `GET http://localhost:9392/v1/models`（Z-Image-Turbo 图像生成服务）

**响应示例：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen-Image-Edit-2509",
      "object": "model",
      "owned_by": "Tongyi-MAI"
    }
  ]
}
```

## API 测试样例

### 测试场景 1：使用 Qwen-Image 生成高质量图像

```bash
curl -X POST http://localhost:9391/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "一只在夕阳下奔跑的金色猎犬，电影级光效",
    "quality": "high",
    "size": "1024x1024",
    "n": 1
  }'
```

**预期输出：** 生成一张高质量图像（50步推理）

---

### 测试场景 2：使用 Z-Image-Turbo 快速生成图像

```bash
curl -X POST http://localhost:9392/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic cityscape at night with neon lights",
    "quality": "medium",
    "size": "1024x768",
    "n": 4
  }'
```

**预期输出：** 快速生成 4 张图像（9步推理）

---

### 测试场景 3：编辑图像添加元素

```bash
# 准备测试图片
curl -X POST http://localhost:9390/v1/images/edits \
  -F "image=@/path/to/input_image.jpg" \
  -F "prompt=在图片右上角添加一轮明月" \
  -F "quality=high" \
  -F "size=1024x1024" \
  -F "n=1"
```

**预期输出：** 返回编辑后的图片（40步推理）

---

### 测试场景 4：批量编辑多张图片

```bash
curl -X POST http://localhost:9390/v1/images/edits \
  -F "image=@/path/to/image1.jpg" \
  -F "image=@/path/to/image2.jpg" \
  -F "prompt=将天空改为橙色" \
  -F "quality=medium" \
  -F "size=1024x768" \
  -F "n=1"
```

**预期输出：** 返回两张编辑后的图片（每张20步推理）

---

### 测试场景 5：低质量快速生成

```bash
curl -X POST http://localhost:9392/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a simple sketch of a tree",
    "quality": "low",
    "size": "512x512",
    "n": 8
  }'
```

**预期输出：** 快速生成 8 张低质量草图（5步推理）

---

### 测试场景 6：查询模型信息

```bash
# 查询 Qwen-Image 服务
curl http://localhost:9391/v1/models

# 查询 Z-Image-Turbo 服务
curl http://localhost:9392/v1/models

# 查询图像编辑服务
curl http://localhost:9390/v1/models
```

---

## 错误处理

### 常见错误及解决方案

| 错误类型 | 可能原因 | 解决方案 |
|---------|---------|---------|
| 模型加载失败 | 模型路径错误或文件不完整 | 检查 `HOST_MODEL_DIR` 路径，重新下载模型 |
| HPU 设备不可用 | Gaudi 驱动未安装或配置错误 | 检查 `hl-smi` 命令，确保驱动正常运行 |
| 内存不足 | 模型超过显存容量 | 增加卡数或使用更小的批次大小 |
| 代理连接失败 | 代理配置错误 | 检查 `http_proxy` 和 `https_proxy` 设置 |
| 容器启动失败 | 端口被占用 | 检查端口 9390/9391/9392 是否被占用 |

### 错误响应格式

```json
{
  "error": {
    "message": "错误描述信息",
    "code": "400"
  }
}
```

## 相关资源

- [Intel Gaudi 官方文档](https://docs.habana.ai/)
- [Optimum Habana GitHub](https://github.com/HabanaAI/optimum-habana-fork)
- [Diffusers 库](https://github.com/huggingface/diffusers)
- [OPEA 项目](https://github.com/opea-project/GenAIComps)
- [项目文档](https://opea-project.github.io/)
