# CosyVoice Docker 构建指南

本文档提供 CosyVoice 服务的独立部署方案。可以提供API形式访问Gaudi上的语音合成模型服务。

本指南提供了基于docker compose的自动部署和命令行的手工部署两种部署方式。

## 主要特性

- **文本转语音**: 支持文本提示和音频条件输入，生成音频。
- **四种生成模式**:
  - **零样本模式**: 音频克隆 + 参考文本，快速模仿特定说话人音色
  - **跨语言模式**: 保留音色特征进行跨语言语音合成（如中文音色生成英文语音）
  - **指令模式**: 通过自然语言指令控制语音风格、情感、方言（最多支持15种指令）
  - **预训练模式**: 使用预设音色（如"Chinese Male"）进行标准TTS合成
- **HPU/Gaudi 优化**: 充分利用 Habana Gaudi 加速器的高性能计算能力。
- **RESTful API**: 提供标准化 RESTful 接口及 OpenAPI 类似接口。
- **容器化部署**: 支持 Docker 快速部署和环境隔离。


## 前置条件

- 已安装 Docker 和 Docker Compose v2
- 可访问 Habana Gaudi Docker 镜像仓库
- 足够的磁盘空间（模型和依赖约需 5GB）

## 硬件资源配置

建议使用1卡部署。

## 安装 Docker Compose v2

```bash
# Ubuntu 22.04 安装命令
sudo apt update
sudo apt install docker-compose-v2
```
如遇到 Docker 版本过低或者版本冲突问题，请查询 Ubuntu 22.04 升级 Docker 和 docker-compose-v2 的文档进行操作。


## 构建 Docker 镜像

```bash
# 克隆 optimum habana fork aice v1.22.0 分支
git clone https://github.com/HabanaAI/optimum-habana-fork.git -b aice/v1.22.0
# 进入 CosyVoice 目录
cd optimum-habana-fork/examples/CosyVoice

# 构建 Docker 镜像
export https_proxy="http://your-proxy-address:port"

docker build --build-arg https_proxy=$https_proxy \
             -f Dockerfile \
             -t cosyvoice:latest .
```

## 下载模型
假设主机模型目录为 ${MODELS_HOST_PATH}, 请在该目录下下载模型，并在后续配置中将该目录映射进入 Docker

```bash
cd ${MODELS_HOST_PATH}
apt install git-lfs
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git 
```

## 运行 Docker 容器

### 方法 1: 使用 Docker Compose（推荐）

使用 Docker Compose 可以简化容器管理，所有配置通过环境变量进行集中管理。

#### 步骤 1: 配置环境变量

根据 env_example 创建 `.env` 文件，放在 docker-compose.yml 相同目录下。
根据您的环境修改以下配置：

```bash
# Habana 设备配置（使用 hl-smi 查看可用设备id,选取空闲设备）
HABANA_VISIBLE_DEVICES=0

# 主机路径配置
MODELS_HOST_PATH=/mnt/disk4/hf_models

# 端口配置
HOST_PORT=8080

```
## 环境变量配置说明

`env_example` 文件包含 CosyVoice 服务运行所需的所有配置参数。

| 变量名 | 说明 | 具体例子 |
|--------|------|----------|
| `HABANA_VISIBLE_DEVICES` | 指定使用的 Habana Gaudi 设备 ID,0-7 任选 | `0` |
| `MODELS_HOST_PATH` | 模型文件在宿主机上的存储路径 | `/mnt/disk4/hf_models` |
| `DATA_HOST_PATH` | 数据文件在宿主机上的存储路径 | `/mnt/disk8` |
| `HOST_PORT` | 宿主机监听端口 | `8080` |

**查看可用设备：**
```bash
hl-smi
```

**路径映射说明：**
- `MODELS_HOST_PATH` → 容器内的 `/models` 目录
- `DATA_HOST_PATH` → 容器内的 `/data` 目录

**端口映射：** `HOST_PORT:9370`


#### 步骤 2: 启动服务

```bash
# 启动容器
docker compose up -f docker-compose.yml -d

# 查看容器日志
docker logs -f cosyvoice-server

# 检查容器状态
docker compose ps
```

#### 步骤 3: 验证服务

```bash
# 检查健康状态

docker logs -f cosyvoice-server
# 出现如下信息表示服务已经正常工作
#=============Application Server Ready to Process Requests.===============

# 测试 API 连接
curl http://localhost:8080/v1/audio/speech \
    -F text="测试文本" \
    -F mode="zero_shot" \
    -F prompt_text="希望你以后能够做的比我还好呦。" \
    -F prompt_audio "@/CosyVoice/asset/zero_shot_prompt.wav"
```

#### 常用管理命令

```bash
# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看实时日志
docker compose logs -f cosyvoice-server

# 进入容器 Shell
docker compose exec cosyvoice-server bash

```

### 方法 2: 使用 Docker Run

如果需要更多自定义配置，可以使用 docker run 命令直接启动：

```bash
docker run -d \
  --name cosyvoice-server \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=0 \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  -p 8080:9370 \
  -v /mnt/disk4/hf_models:/models \
  -v /mnt/disk8:/data \
  -v /dev:/dev \
  --cap-add=SYS_NICE \
  --cap-add=IPC_LOCK \
  --device=/dev:/dev \
  cosyvoice:latest \
  /bin/bash /CosyVoice/start.sh
```

## API 使用示例

服务启动后，API 服务器地址为 `http://localhost:8080`（或您配置的 HOST_PORT）

### 零样本示例

```bash
curl http://localhost:8080/v1/audio/speech \
    -F text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
    -F mode="zero_shot" \
    -F prompt_text="希望你以后能够做的比我还好呦。" \
    -F prompt_audio "@/CosyVoice/asset/zero_shot_prompt.wav"
```

### 跨语言示例

```bash
curl http://localhost:8080/v1/audio/speech \
    -F text="If one knows how to be grateful and content with small things, then he is a happy person." \
    -F mode="cross_lingual" \
    -F prompt_audio "@/CosyVoice/asset/cross_lingual_prompt.wav"
```

### 指令示例

```bash
curl http://localhost:8080/v1/audio/speech \
    -F text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。" \
    -F mode="instruct" \
    -F instruct_text="用四川话说这句话" \
    -F prompt_audio="@/CosyVoice/asset/cross_lingual_prompt.wav"
```

### 预训练示例

```bash
curl http://localhost:8080/v1/audio/speech \
    -F text="这次机会让我能够在新的领域中不断学习和成长，同时也激励我去克服自身的不足。" \
    -F mode="pretrain" \
    -F pretrained_tone="Chinese Male"
```

## 故障排除

### 查看容器日志

```bash
docker logs cosyvoice-server
```

### 进入容器 Shell

```bash
docker exec -it cosyvoice-server bash
```

### 停止容器

```bash
docker stop cosyvoice-server
docker rm cosyvoice-server
```
