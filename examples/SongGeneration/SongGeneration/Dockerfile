# SongGeneration API Server Dockerfile
# Based on Habana Gaudi PyTorch image

FROM vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PT_HPU_LAZY_MODE=1
# Proxy server config is optional 
#ENV https_proxy=http://child-prc.intel.com:912

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-gaudi.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-gaudi.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    opea-comps==1.5 \
    huggingface-hub \
    fastapi \
    uvicorn \
    soundfile \
    pytz \
    pydantic

# Copy application code
COPY . .

# Copy runtime from tools directory to the correct locations
# Runtime contains ckpt and third_party subdirectories
RUN if [ -d "tools/runtime" ]; then \
        echo "Copying runtime files..."; \
        if [ -d "tools/runtime/ckpt" ]; then \
            mkdir -p ckpt && cp -r tools/runtime/ckpt/* ckpt/ 2>/dev/null || true; \
        fi; \
        if [ -d "tools/runtime/third_party" ]; then \
            mkdir -p third_party && cp -r tools/runtime/third_party/* third_party/ 2>/dev/null || true; \
        fi; \
        echo "Runtime files copied successfully"; \
    else \
        echo "Warning: tools/runtime directory not found"; \
    fi

# Create necessary directories for mounted volumes
RUN mkdir -p /app/models /app/logs /app/data

# Set Python path
ENV PYTHONPATH="/app/codeclm/tokenizer/:/app:/app/codeclm/tokenizer/Flow1dVAE/:/app/codeclm/tokenizer/:$PYTHONPATH"

# Expose the default port (can be overridden)
EXPOSE 8485

# Entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Use entrypoint script to handle parameter passing
ENTRYPOINT ["/app/docker-entrypoint.sh"]
