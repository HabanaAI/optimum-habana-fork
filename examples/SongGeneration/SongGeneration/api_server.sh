#!/bin/bash
# SongGeneration API Server Startup Script
# Usage: ./api_server.sh [options]
#
# Options:
#   -p, --port PORT          Server port (default: 8485)
#   -m, --model-dir DIR      Model directory path (default: songgeneration_base_new)
#   -l, --log-dir DIR        Log directory path (default: ./logs)
#   -f, --flash-attn         Enable flash attention
#   -h, --help              Show this help message
#
# Environment Variables:
#   SERVER_PORT             Server port
#   MODEL_DIR               Model directory name
#   LOG_DIR                 Log directory path
#   USE_FLASH_ATTN          Enable flash attention (true/false)
#
# Examples:
#   ./api_server.sh                                    # Use defaults
#   ./api_server.sh -p 8080                            # Custom port
#   ./api_server.sh -m songgeneration_large            # Use large model
#   ./api_server.sh -p 8080 -m songgeneration_large -l /var/log/songgeneration
#   ./api_server.sh --port 8080 --model-dir songgeneration_large --flash-attn

set -e

# Default values
DEFAULT_PORT=8485
DEFAULT_MODEL_DIR="songgeneration_base_new"
DEFAULT_LOG_DIR="./logs"
DEFAULT_USE_FLASH_ATTN=false

# Parse command line arguments
PORT=""
MODEL_DIR=""
LOG_DIR=""
USE_FLASH_ATTN=""

show_help() {
    grep "^#" "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -f|--flash-attn)
            USE_FLASH_ATTN="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Use environment variables or defaults if not set via command line
SERVER_PORT=${PORT:-${SERVER_PORT:-$DEFAULT_PORT}}
MODEL_PATH=${MODEL_DIR:-${MODEL_DIR:-$DEFAULT_MODEL_DIR}}
LOG_PATH=${LOG_DIR:-${LOG_DIR:-$DEFAULT_LOG_DIR}}
FLASH_ATTN=${USE_FLASH_ATTN:-${USE_FLASH_ATTN:-$DEFAULT_USE_FLASH_ATTN}}

# Convert to boolean string for flash attention
if [ "$FLASH_ATTN" = "true" ] || [ "$FLASH_ATTN" = "True" ] || [ "$FLASH_ATTN" = "1" ]; then
    FLASH_ATTN_STR="true"
else
    FLASH_ATTN_STR="false"
fi

# Validate model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory '$MODEL_PATH' not found!"
    echo "Please ensure the model directory exists or download the model:"
    echo "  huggingface-cli download lglg666/SongGeneration-base-new --local-dir ./songgeneration_base_new"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_PATH"

# Set log file path with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_PATH/api_server_${TIMESTAMP}.log"

echo "========================================"
echo "SongGeneration API Server Startup"
echo "========================================"
echo "Server Port: $SERVER_PORT"
echo "Model Directory: $MODEL_PATH"
echo "Log Directory: $LOG_PATH"
echo "Log File: $LOG_FILE"
echo "Flash Attention: $FLASH_ATTN_STR"
echo "========================================"
echo ""

# Check for required files
if [ ! -f "tools/new_prompt.pt" ]; then
    echo "WARNING: tools/new_prompt.pt not found!"
    echo "Please download it:"
    echo "  wget https://media.githubusercontent.com/media/tencent-ailab/SongGeneration/refs/heads/main/tools/new_prompt.pt"
    echo "  mv new_prompt.pt tools/"
    echo ""
fi

# Set environment variables
export PYTHONPATH="$(pwd)/codeclm/tokenizer/:$(pwd):$(pwd)/codeclm/tokenizer/Flow1dVAE/:$(pwd)/codeclm/tokenizer/:$PYTHONPATH"
export PT_HPU_LAZY_MODE=1

# Modify api_server.py to use the specified model directory
# We need to update the hardcoded path in api_server.py
API_SERVER_FILE="api_server.py"
if [ -f "$API_SERVER_FILE" ]; then
    # Check if we need to backup the original file
    if [ ! -f "${API_SERVER_FILE}.bak" ]; then
        cp "$API_SERVER_FILE" "${API_SERVER_FILE}.bak"
        echo "Created backup: ${API_SERVER_FILE}.bak"
    fi
    
    # Update the model path in api_server.py
    echo "Updating model path in $API_SERVER_FILE to: $MODEL_PATH"
    sed -i "s|ckpt_path=\"songgeneration_base_new\"|ckpt_path=\"$MODEL_PATH\"|g" "$API_SERVER_FILE"
    sed -i "s|ckpt_path=\"[^\"]*\"|ckpt_path=\"$MODEL_PATH\"|g" "$API_SERVER_FILE"
fi

echo "Starting API server..."
echo "Command: PT_HPU_LAZY_MODE=1 python api_server.py --server-port $SERVER_PORT"
echo "Logs will be written to: $LOG_FILE"
echo ""

# Start the server with logging
PT_HPU_LAZY_MODE=1 python api_server.py --server-port "$SERVER_PORT" 2>&1 | tee "$LOG_FILE"
