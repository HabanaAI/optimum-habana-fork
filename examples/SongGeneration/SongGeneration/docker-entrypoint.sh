#!/bin/bash
# Docker entrypoint script for SongGeneration API Server

# Default values
DEFAULT_PORT=8485
DEFAULT_MODEL_DIR="songgeneration_base_new"
DEFAULT_LOG_DIR="/app/logs"
DEFAULT_USE_FLASH_ATTN="false"

# Get values from environment variables or use defaults
SERVER_PORT=${SERVER_PORT:-$DEFAULT_PORT}
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-$DEFAULT_USE_FLASH_ATTN}

# Validate model directory
if [ ! -d "/app/models/$MODEL_DIR" ]; then
    echo "ERROR: Model directory '/app/models/$MODEL_DIR' not found!"
    echo "Available models in /app/models:"
    ls -la /app/models/ 2>/dev/null || echo "  (directory is empty or not mounted)"
    exit 1
fi

echo "========================================"
echo "SongGeneration API Server Configuration"
echo "========================================"
echo "Server Port: $SERVER_PORT"
echo "Model Directory: /app/models/$MODEL_DIR"
echo "Log Directory: $LOG_DIR"
echo "Use Flash Attention: $USE_FLASH_ATTN"
echo "========================================"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set log file path
LOG_FILE="$LOG_DIR/api_server.log"

echo "Log file: $LOG_FILE"
echo "Starting server..."
echo ""

# Update the model path in api_model.py to use the mounted models directory
# This is a workaround since api_model.py has a hardcoded path
export MODEL_PATH="/app/models/$MODEL_DIR"

# Modify api_server.py to accept model directory parameter
# We'll create a modified version temporarily
if [ -f "api_server.py" ]; then
    # Check if we need to modify the hardcoded path
    if grep -q 'ckpt_path="songgeneration_base_new"' api_server.py; then
        echo "Modifying api_server.py to use model path: $MODEL_PATH"
        sed -i "s|ckpt_path=\"songgeneration_base_new\"|ckpt_path=\"$MODEL_PATH\"|g" api_server.py
    fi
fi

# Check if new_prompt.pt exists in tools directory
if [ ! -f "tools/new_prompt.pt" ]; then
    echo "WARNING: tools/new_prompt.pt not found!"
    echo "Please ensure the file is mounted to /app/tools/new_prompt.pt"
fi

# Build command arguments
FLASH_ATTN_ARG=""
if [ "$USE_FLASH_ATTN" = "true" ] || [ "$USE_FLASH_ATTN" = "True" ]; then
    # Note: The api_model.py expects use_flash_attn parameter
    # This would require modifying api_server.py to pass this parameter
    echo "Note: Flash attention setting requires api_server.py modification"
fi

# Start the server with logging
export PYTHONPATH="/app/codeclm/tokenizer/:/app:/app/codeclm/tokenizer/Flow1dVAE/:/app/codeclm/tokenizer/:$PYTHONPATH"
export PT_HPU_LAZY_MODE=1

# Run the server
python api_server.py --server-port "$SERVER_PORT" 2>&1 | tee "$LOG_FILE"
