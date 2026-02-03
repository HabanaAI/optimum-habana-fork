#!/bin/bash
# Quick start script for SongGeneration Docker deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    cat << EOF
SongGeneration Docker Quick Start Script

Usage: $0 [command]

Commands:
    setup       - Prepare directories and check prerequisites
    build       - Build Docker image
    start       - Start the service
    stop        - Stop the service
    restart     - Restart the service
    logs        - View service logs
    status      - Check service status
    test        - Test API endpoints
    clean       - Remove containers and images
    help        - Show this help message

Examples:
    $0 setup    # Initial setup
    $0 build    # Build the Docker image
    $0 start    # Start the service
    $0 logs     # View logs

EOF
}

setup() {
    echo "========================================"
    echo "Setting up SongGeneration Docker environment"
    echo "========================================"
    
    # Create necessary directories
    echo "Creating directories..."
    mkdir -p models tools logs data
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        echo "Creating .env file from example..."
        cp env.example .env
        echo "✓ Created .env file. Please edit it with your configuration."
    else
        echo "✓ .env file already exists"
    fi
    
    # Check for required files
    echo ""
    echo "Checking required files..."
    
    if [ ! -f "tools/new_prompt.pt" ]; then
        echo "⚠ tools/new_prompt.pt not found!"
        echo "  Download it with:"
        echo "  wget https://media.githubusercontent.com/media/tencent-ailab/SongGeneration/refs/heads/main/tools/new_prompt.pt"
        echo "  mv new_prompt.pt tools/"
    else
        echo "✓ tools/new_prompt.pt found"
    fi
    
    # Check for runtime
    if [ ! -d "tools/runtime" ]; then
        echo "⚠ tools/runtime not found!"
        echo "  Download it with:"
        echo "  huggingface-cli download lglg666/SongGeneration-Runtime --local-dir ./tools/runtime"
    else
        echo "✓ tools/runtime found"
    fi
    
    # Check for models
    echo ""
    echo "Checking models..."
    MODEL_COUNT=$(find models -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        echo "⚠ No models found in ./models directory!"
        echo "  Download models with:"
        echo "  huggingface-cli download lglg666/SongGeneration-base-new --local-dir ./models/songgeneration_base_new"
    else
        echo "✓ Found $MODEL_COUNT model(s)"
        ls -1 models/
    fi
    
    echo ""
    echo "========================================"
    echo "Setup complete!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env file with your configuration"
    echo "  2. Download required models if not already done"
    echo "  3. Run: $0 build"
    echo "  4. Run: $0 start"
}

build() {
    echo "Building Docker image..."
    # https_proxy config is optional which depends on network config
    #export https_proxy=http://child-prc.intel.com:912
    docker-compose build 
    echo "✓ Build complete"
}

start() {
    echo "Starting SongGeneration API service..."
    docker-compose up -d
    echo "✓ Service started"
    echo ""
    echo "API available at: http://localhost:$(grep SERVER_PORT .env | cut -d= -f2 | tr -d ' ')"
    echo "View logs with: $0 logs"
}

stop() {
    echo "Stopping SongGeneration API service..."
    docker-compose down
    echo "✓ Service stopped"
}

restart() {
    echo "Restarting SongGeneration API service..."
    docker-compose restart
    echo "✓ Service restarted"
}

logs() {
    docker-compose logs -f
}

status() {
    echo "Service status:"
    docker-compose ps
    echo ""
    echo "Health check:"
    PORT=$(grep SERVER_PORT .env | cut -d= -f2 | tr -d ' ')
    if curl -s "http://localhost:$PORT/v1/audio/song/query/auto_prompt_audio_type" > /dev/null 2>&1; then
        echo "✓ API is responding"
    else
        echo "✗ API is not responding"
    fi
}

test_api() {
    PORT=$(grep SERVER_PORT .env | cut -d= -f2 | tr -d ' ')
    echo "Testing API at http://localhost:$PORT"
    echo ""
    
    echo "Test 1: Query auto_prompt_audio_type"
    curl -s "http://localhost:$PORT/v1/audio/song/query/auto_prompt_audio_type" | head -c 200
    echo ""
    echo ""
    
    echo "Test 2: Create generation task"
    RESPONSE=$(curl -s -X POST "http://localhost:$PORT/v1/audio/song" \
        --form-string gt_lyric="[intro-short] ; [verse] Test lyrics. ; [outro-short]")
    echo "$RESPONSE"
    echo ""
}

clean() {
    echo "This will remove containers and images. Are you sure? (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        docker-compose down --rmi all --volumes
        echo "✓ Cleanup complete"
    else
        echo "Cancelled"
    fi
}

# Main command handler
case "${1:-help}" in
    setup)
        setup
        ;;
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    test)
        test_api
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
