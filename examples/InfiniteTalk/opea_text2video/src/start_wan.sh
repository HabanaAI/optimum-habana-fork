#!/bin/bash
# start.sh - 简化版启动脚本

# 创建日志目录
NUM_CARDS=${HABANA_USED_DEVICE}
echo "Using HPU $NUM_CARDS cards"
mkdir -p /hf/logs

echo "=== 启动 Text2Video 服务 ==="
echo ""

# 启动 web 服务
echo "启动 Web 服务..."
python3 web_service.py \
  --model_name_or_path Wan2.2-TI2V-5B  \
  --rank_size ${NUM_CARDS}  \
  --video_dir /home/user/video > /hf/logs/web.log 2>&1 &
WEB_PID=$!
echo $WEB_PID > /tmp/web.pid
echo "Web 服务已启动 (PID: $WEB_PID)"

# 等待 web 服务启动
echo "等待 Web 服务初始化..."
sleep 20

# 启动 job 服务
echo "启动 Job 服务..."
PT_HPU_RECIPE_CACHE_CONFIG=/home/user/cache_wan,false,40960 PT_HPU_SYNC_LAUNCH=1 PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 torchrun \
  --nproc_per_node=${NUM_CARDS} \
  --master-port 29502 \
  --standalone \
  job_service_wan.py \
  --ulysses_size=${NUM_CARDS} > /hf/logs/job.log 2>&1 &
JOB_PID=$!
echo "Job 服务已启动 (PID: $JOB_PID)"

echo "Waiting for model loading, may take 3~5 mins"
sleep 300
echo $JOB_PID > /tmp/job.pid
echo ""
echo "=== 服务状态 ==="
echo "Web 服务: 运行中 (PID: $WEB_PID)"
echo "Job 服务: 运行中 (PID: $JOB_PID)"
echo "==============="

echo ""
echo "日志文件:"
echo "Web 服务: /hf/logs/web.log"
echo "Job 服务: /hf/logs/job.log"
echo ""
echo "容器保持运行，按 Ctrl+C 停止"
echo "使用 docker exec -it <容器名> bash 进入容器"

# 保持容器运行
tail -f /dev/null
