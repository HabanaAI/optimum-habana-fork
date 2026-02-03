# SongGeneration Docker 部署指南

## 概述

本文档详细说明如何使用 Docker 部署 SongGeneration API 服务，支持在 Habana Gaudi 硬件上运行。

## 前提条件

- Docker Engine 20.10+
- Docker Compose 2.0+
- Habana Gaudi 设备驱动程序
- 约 40GB 磁盘空间用于模型文件

## 目录结构设置

### 1. 创建必要的目录

在当前项目根目录下创建以下目录结构：

```bash
mkdir -p models tools logs data
```

目录说明：
- `models/` - 存放模型权重文件
- `tools/` - 存放 new_prompt.pt 和 runtime 文件
- `logs/` - 容器日志持久化目录
- `data/` - 生成的音频文件输出目录

### 2. 完整目录结构

```
SongGeneration/
├── docker-compose.yml          # Docker Compose 配置文件
├── env.example                # 环境变量模板
├── models/                     # 模型权重目录
│   ├── songgeneration_base/
│   ├── songgeneration_base_new/
│   ├── songgeneration_base_full/
│   └── songgeneration_large/
├── tools/                      # 工具文件目录
│   ├── new_prompt.pt          # 必需：提示词文件
│   └── runtime/               # Runtime 依赖（自动复制到容器）
│       ├── ckpt/
│       └── third_party/
├── logs/                      # 日志文件目录（持久化）
└── data/                      # 生成的音频输出目录

SongGeneration/SongGeneration/  # 应用程序代码目录
├── Dockerfile                 # Docker 镜像构建文件
├── docker-entrypoint.sh       # 容器入口脚本
├── api_server.sh             # API 服务启动脚本
└── ...                       # 其他应用程序文件
```

## 模型文件下载

### 必需文件

#### 1. new_prompt.pt

```bash
wget https://media.githubusercontent.com/media/tencent-ailab/SongGeneration/refs/heads/main/tools/new_prompt.pt
mv new_prompt.pt tools/
```

#### 2. Runtime 依赖

```bash
huggingface-cli download lglg666/SongGeneration-Runtime --local-dir ./tools/runtime
```

Runtime 包含以下子目录，将在 Docker build 时自动复制到容器内：
- `ckpt/` - 检查点文件
- `third_party/` - 第三方依赖（如 demucs）

#### 3. 模型权重（选择需要的模型）

**基础模型（推荐）**：
```bash
huggingface-cli download lglg666/SongGeneration-base-new --local-dir ./models/songgeneration_base_new
```

**其他可选模型**：
```bash
# 基础版本
huggingface-cli download lglg666/SongGeneration-base --local-dir ./models/songgeneration_base

# 完整版本（更大，效果更好）
huggingface-cli download lglg666/SongGeneration-base-full --local-dir ./models/songgeneration_base_full

# 大模型版本（最大，最佳效果）
huggingface-cli download lglg666/SongGeneration-large --local-dir ./models/songgeneration_large
```

### 模型对比

| 模型 | 大小 | 推荐用途 | 显存需求 |
|------|------|----------|----------|
| songgeneration_base | ~2GB | 快速测试 | 16GB |
| songgeneration_base_new | ~2GB | **推荐用于生产** | 16GB |
| songgeneration_base_full | ~4GB | 更好的质量 | 24GB |
| songgeneration_large | ~8GB | 最佳质量 | 32GB+ |

## 环境变量配置 (.env)

### 1. 创建环境配置文件

```bash
cp env.example .env
```

### 2. 环境变量说明

编辑 `.env` 文件，配置以下参数：

#### 服务器配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SERVER_PORT` | 8485 | API 服务端口 |
| `MODEL_DIR` | songgeneration_base_new | 使用的模型目录名 |
| `USE_FLASH_ATTN` | false | 是否使用 Flash Attention |
| `LOG_LEVEL` | INFO | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

#### Habana 设备配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `HABANA_VISIBLE_DEVICES` | all 或者id  | 可见的 Habana 设备 |

设置值：
- `all` - 使用所有可用的 Habana 设备
- `0` - 仅使用第一个设备,id 可以是0-7 任意一个数值

#### 路径配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MODELS_HOST_PATH` | ./models | 主机模型目录路径 |
| `LOGS_HOST_PATH` | ./logs | 主机日志目录路径 |
| `TOOLS_HOST_PATH` | ./tools | 主机工具目录路径 |
| `DATA_HOST_PATH` | ./data | 主机数据目录路径 |

**注意**：路径可以是相对路径（相对于 docker-compose.yml）或绝对路径。

#### 资源配置（可选）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MEMORY_LIMIT` | 32G | 容器内存限制 |

### 3. 示例 .env 配置

```env
# 服务器配置
SERVER_PORT=8485
MODEL_DIR=songgeneration_base_new
USE_FLASH_ATTN=false
LOG_LEVEL=INFO

# Habana 设备配置
HABANA_VISIBLE_DEVICES=all

# 路径配置（使用绝对路径示例）
MODELS_HOST_PATH=/data/models
LOGS_HOST_PATH=/data/logs
TOOLS_HOST_PATH=/data/tools
DATA_HOST_PATH=/data/outputs

# 资源限制
MEMORY_LIMIT=64G
```

## Docker Compose 配置

### docker-compose.yml 关键配置说明

```yaml
version: '3.8'

services:
  songgeneration-api:
    build:
      context: ./SongGeneration
      dockerfile: Dockerfile
    image: songgeneration-api:latest
    container_name: songgeneration-api
    runtime: habana  # Habana Gaudi 运行时
    command: /bin/bash /app/api_server.sh -p ${SERVER_PORT:-8485} -m /app/models/${MODEL_DIR:-songgeneration_base_new} -l /app/logs
    
    environment:
      # 所有环境变量从 .env 文件读取
      - SERVER_PORT=${SERVER_PORT:-8485}
      - HABANA_VISIBLE_DEVICES=${HABANA_VISIBLE_DEVICES:-all}
      - OMPI_MCA_btl_vader_single_copy_mechanism=none  # MPI 配置
      
    ports:
      - "${SERVER_PORT:-8485}:${SERVER_PORT:-8485}"  # 端口映射
    
    volumes:
      - ${MODELS_HOST_PATH:-./models}:/app/models:ro     # 模型（只读）
      - ${LOGS_HOST_PATH:-./logs}:/app/logs              # 日志
      - ${TOOLS_HOST_PATH:-./tools}:/app/tools:ro        # 工具（只读）
      - ${DATA_HOST_PATH:-./data}:/app/data              # 数据输出
      - /dev:/dev                                        # 设备访问
    
    devices:
      - /dev:/dev  # Habana 设备访问
    
    cap_add:
      - SYS_NICE   # 进程优先级调整
      - IPC_LOCK   # 内存锁定
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:${SERVER_PORT:-8485}/v1/audio/song/query/auto_prompt_audio_type"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 180s  # 3分钟后开始检查（给模型加载时间）
      start_interval: 10s
```

### 权限说明

| 配置 | 值 | 说明 |
|------|-----|------|
| `runtime` | habana | 使用 Habana Gaudi 运行时 |
| `cap_add: SYS_NICE` | - | 允许调整进程优先级 |
| `cap_add: IPC_LOCK` | - | 允许锁定内存（防止交换） |
| `devices: /dev:/dev` | - | 访问主机设备树 |
| `volumes: /dev:/dev` | - | 挂载设备目录 |

## 部署步骤

### 方法一：使用快速启动脚本（推荐）

#### 1. 安装 Docker Compose（如未安装）
具体步骤参考DeepSeek 即可
```bash

# 验证安装
docker-compose --version
```

#### 2. 初始化环境

```bash
# 创建目录、检查文件、生成 .env
./docker-start.sh setup
```

此命令会：
- 创建 models、tools、logs、data 目录
- 检查必需的 new_prompt.pt 和 runtime 文件
- 检查模型文件
- 生成 .env 配置文件（如果不存在）

#### 3. 编辑配置文件

```bash
vim .env  # 或使用其他编辑器
```

根据需要修改端口、模型路径、设备配置等。

#### 4. 构建 Docker 镜像

```bash
./docker-start.sh build
```

此命令会：
- 设置代理环境变量
- 使用 --build-arg 传递代理
- 构建 Docker 镜像


#### 5. 启动服务

```bash
./docker-start.sh start
```

#### 6. 检查服务状态

```bash
# 查看容器状态
./docker-start.sh status

# 查看日志
./docker-start.sh logs

# 测试 API
./docker-start.sh test
```

### 方法二：手动部署

#### 1. 准备环境

```bash
# 创建目录
mkdir -p models tools logs data

# 下载必需文件
wget https://media.githubusercontent.com/media/tencent-ailab/SongGeneration/refs/heads/main/tools/new_prompt.pt -O tools/new_prompt.pt
huggingface-cli download lglg666/SongGeneration-Runtime --local-dir ./tools/runtime
huggingface-cli download lglg666/SongGeneration-base-new --local-dir ./models/songgeneration_base_new

# 配置环境变量
cp env.example .env
# 编辑 .env 文件
```

#### 2. 构建并启动

```bash
# 设置代理,可选配置
export https_proxy=http://your_proxy_server:port

# 构建
docker-compose build --build-arg https_proxy=$https_proxy

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## API 使用

### 服务启动后，API 端点

- **健康检查**：`GET http://localhost:8485/v1/audio/song/query/auto_prompt_audio_type`
- **生成音频**：`POST http://localhost:8485/v1/audio/song`
- **查询状态**：`GET http://localhost:8485/v1/audio/song/{task_id}`
- **下载音频**：`GET http://localhost:8485/v1/audio/song/{task_id}/content`

### 示例请求

```bash
# 查询支持的音频类型
curl http://localhost:8485/v1/audio/song/query/auto_prompt_audio_type

# 生成音频
curl -X POST http://localhost:8485/v1/audio/song \
    --form-string gt_lyric="[intro-short] ; [verse] Test lyrics. ; [outro-short]"
```

## 常用命令

```bash
# 构建
docker-compose build

# 启动
docker-compose up -d

# 停止
docker-compose down

# 重启
docker-compose restart

# 查看日志
docker-compose logs -f

# 进入容器
docker exec -it songgeneration-api bash

# 检查容器状态
docker-compose ps

# 删除容器和镜像
docker-compose down --rmi all
```

## 参考文档

- 项目 README.md - API 详细文档和使用示例
- SongGeneration/Dockerfile - 镜像构建详情

## 支持

如遇问题：
1. 查看容器日志：`docker-compose logs`
2. 检查配置文件：`.env`
3. 验证文件完整性：模型、tools、runtime
