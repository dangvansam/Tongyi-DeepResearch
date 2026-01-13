#!/bin/bash

# =============================================================================
# vLLM Docker Compose Deployment Script
# Uses Docker Compose for reliable deployment with proper CUDA support
# Single service approach - model specified via environment variables
# =============================================================================

set -e

# Default values
MODEL=""
SERVED_MODEL_NAME=""
ACTION="up"
GPU_DEVICES="1"
TENSOR_PARALLEL_SIZE=""
PORT=8000
DETACH=true
MAX_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.9
KV_CACHE_DTYPE=""
MAX_MODEL_LEN=""
EXTRA_ARGS=""

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.vllm.yml"

# Model presets: model_id|tensor_parallel|kv_cache_dtype|mode
# Served model name will be derived from model_id automatically
declare -A MODEL_PRESETS=(
    # LLaMA models
    ["llama70b"]="meta-llama/Llama-3.3-70B-Instruct|2||non-thinking"
    ["llama70b-fp8"]="nvidia/Llama-3.3-70B-Instruct-FP8|1|fp8|non-thinking"
    ["llama70b-fp4"]="nvidia/Llama-3.3-70B-Instruct-FP4|1|fp8|non-thinking"
    ["llama8b"]="meta-llama/Llama-3.1-8B-Instruct|1||non-thinking"
    ["llama8b-fp8"]="nvidia/Llama-3.1-8B-Instruct-FP8|1|fp8|non-thinking"
    # Qwen3-Next models
    ["qwen3-next-80b"]="Qwen/Qwen3-Next-80B-A3B-Instruct|1||non-thinking"
    ["qwen3-next-80b-fp8"]="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8|1||non-thinking"
    ["qwen3-next-80b-fp4"]="RESMP-DEV/Qwen3-Next-80B-A3B-Instruct-NVFP4|1||non-thinking"
    ["qwen3-next-80b-thinking-fp8"]="Qwen/Qwen3-Next-80B-A3B-Thinking-FP8|1||thinking"
    # Qwen3 models
    ["qwen3-235b"]="Qwen/Qwen3-235B-A22B-Instruct|4||non-thinking"
    ["qwen3-235b-fp8"]="Qwen/Qwen3-235B-A22B-Instruct-FP8|4||non-thinking"
    ["qwen3-32b"]="Qwen/Qwen3-32B|1||non-thinking"
    ["qwen3-32b-fp8"]="Qwen/Qwen3-32B-FP8|1|fp8|non-thinking"
    ["qwen3-32b-fp8-thinking"]="Qwen/Qwen3-32B-FP8|1|fp8|thinking"
    ["qwen3-32b-fp8-nothink"]="Qwen/Qwen3-32B-FP8|1|fp8|non-thinking"
    ["qwen3-30b-2507-thinking-fp8"]="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8|1||thinking"
    # Other models
    ["tongyi-deepresearch"]="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B|1||non-thinking"
    ["deepresearch"]="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B|1||non-thinking"
)

print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS] [ACTION] <model>"
    echo ""
    echo -e "${BLUE}Actions:${NC}"
    echo "  up                    Start the service (default)"
    echo "  down                  Stop the service"
    echo "  logs                  View logs"
    echo "  restart               Restart the service"
    echo "  ps                    Show running services"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -m, --model           HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B-Instruct')"
    echo "  -n, --name            Served model name (default: auto from model)"
    echo "  -g, --gpus            GPU devices (e.g., '0', '0,1', '0,1,2,3') (default: 0)"
    echo "  -t, --tensor-parallel Tensor parallel size (default: auto based on GPUs)"
    echo "  -p, --port            Server port (default: 8000)"
    echo "  --max-batch-tokens    Max batched tokens (default: 8192)"
    echo "  --gpu-util            GPU memory utilization ratio 0.0-1.0 (default: 0.9)"
    echo "  --kv-cache-dtype      KV cache dtype (e.g., 'fp8', 'auto')"
    echo "  --max-model-len       Maximum model context length (default: auto)"
    echo "  --extra-args          Additional vLLM arguments (quoted string)"
    echo "  --foreground          Run in foreground (show logs)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo -e "${BLUE}Model Presets:${NC}"
    echo "  LLaMA Models:"
    echo "    llama70b            LLaMA 3.3 70B (BF16, ~140GB VRAM)"
    echo "    llama70b-fp8        LLaMA 3.3 70B FP8 (~70GB VRAM)"
    echo "    llama70b-fp4        LLaMA 3.3 70B FP4 (~35GB VRAM, Blackwell only)"
    echo "    llama8b             LLaMA 3.1 8B (BF16)"
    echo "    llama8b-fp8         LLaMA 3.1 8B FP8"
    echo ""
    echo "  Qwen Models:"
    echo "    qwen3-next-80b      Qwen3-Next 80B (BF16)"
    echo "    qwen3-next-80b-fp8  Qwen3-Next 80B FP8"
    echo "    qwen3-next-80b-fp4  Qwen3-Next 80B NVFP4 (Blackwell only)"
    echo "    qwen3-next-80b-thinking-fp8  Qwen3-Next 80B Thinking FP8 (262K, speculative)"
    echo "    qwen3-235b          Qwen3 235B (BF16)"
    echo "    qwen3-235b-fp8      Qwen3 235B FP8"
    echo "    qwen3-32b           Qwen3 32B (BF16)"
    echo "    qwen3-32b-fp8       Qwen3 32B FP8"
    echo "    qwen3-32b-fp8-thinking   Qwen3 32B FP8 with reasoning (131K context)"
    echo "    qwen3-32b-fp8-nothink    Qwen3 32B FP8 non-thinking (131K context)"
    echo "    qwen3-30b-2507-thinking-fp8  Qwen3-30B-A3B Thinking 2507 FP8 (262K context)"
    echo ""
    echo "  Other Models:"
    echo "    tongyi-deepresearch Tongyi-DeepResearch 30B"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  # Start using a preset"
    echo "  $0 -g 1 llama70b-fp8"
    echo ""
    echo "  # Start with custom model"
    echo "  $0 -g 0 -m mistralai/Mistral-7B-Instruct-v0.3 -n mistral-7b"
    echo ""
    echo "  # Start with extra vLLM args"
    echo "  $0 -g 0,1 --extra-args '--enable-prefix-caching' qwen3-next-fp8"
    echo ""
    echo "  # Set GPU memory utilization"
    echo "  $0 -g 0 --gpu-util 0.7 llama8b"
    echo ""
    echo "  # Stop the service"
    echo "  $0 down"
    echo ""
    echo "  # View logs"
    echo "  $0 logs"
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -n|--name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_DEVICES="$2"
            shift 2
            ;;
        -t|--tensor-parallel)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --max-batch-tokens)
            MAX_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --gpu-util)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --extra-args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --foreground)
            DETACH=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        up|down|logs|restart|ps)
            ACTION="$1"
            shift
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Get preset from positional args if provided
PRESET=""
if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    PRESET="${POSITIONAL_ARGS[0]}"
fi

# Apply preset if specified
if [[ -n "$PRESET" && -n "${MODEL_PRESETS[$PRESET]}" ]]; then
    echo -e "${GREEN}Using preset: $PRESET${NC}"
    IFS='|' read -r preset_model preset_tp preset_kv preset_mode <<< "${MODEL_PRESETS[$PRESET]}"
    MODEL="${MODEL:-$preset_model}"
    # Use raw model name as served model name (extract from path)
    if [[ -z "$SERVED_MODEL_NAME" ]]; then
        SERVED_MODEL_NAME=$(basename "$preset_model")
    fi
    if [[ -z "$TENSOR_PARALLEL_SIZE" ]]; then
        TENSOR_PARALLEL_SIZE="$preset_tp"
    fi
    if [[ -z "$KV_CACHE_DTYPE" && -n "$preset_kv" ]]; then
        KV_CACHE_DTYPE="$preset_kv"
    fi

    # Handle thinking/non-thinking modes
    if [[ "$preset_mode" == "thinking" ]]; then
        # Thinking mode: enable reasoning
        EXTRA_ARGS="${EXTRA_ARGS} --reasoning-parser deepseek_r1"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
        echo -e "${BLUE}Mode:${NC}            Thinking (reasoning enabled)"
    elif [[ "$preset_mode" == "non-thinking" ]]; then
        # Non-thinking mode: no reasoning enabled
        echo -e "${BLUE}Mode:${NC}            Non-thinking"
    fi

    # Enable FP8/FP4 optimizations for quantized models
    if [[ "$PRESET" == *"-fp8"* ]]; then
        export VLLM_USE_FLASHINFER_MOE_FP8=1
    elif [[ "$PRESET" == *"-fp4"* ]]; then
        export VLLM_USE_FLASHINFER_MOE_FP4=1
    fi
elif [[ -n "$PRESET" && "$ACTION" == "up" ]]; then
    echo -e "${RED}Error: Unknown preset '$PRESET'${NC}"
    echo "Use -m to specify a custom model or choose from available presets."
    print_usage
    exit 1
fi

# Validate model for 'up' action
if [[ "$ACTION" == "up" && -z "$MODEL" ]]; then
    echo -e "${RED}Error: Model is required${NC}"
    echo "Specify a preset name or use -m to provide a HuggingFace model ID."
    print_usage
    exit 1
fi

# Auto-detect tensor parallel size from GPU count if not specified
if [[ -z "$TENSOR_PARALLEL_SIZE" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
    TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}
fi

# Auto-generate served model name if not specified
if [[ -z "$SERVED_MODEL_NAME" && -n "$MODEL" ]]; then
    # Extract model name from path (e.g., "meta-llama/Llama-3.1-8B-Instruct" -> "Llama-3.1-8B-Instruct")
    SERVED_MODEL_NAME=$(basename "$MODEL")
fi

# Check for Docker Compose file
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo -e "${RED}Error: docker-compose.vllm.yml not found at $COMPOSE_FILE${NC}"
    exit 1
fi

# Build vLLM command arguments
build_vllm_args() {
    local args="--model $MODEL"
    args+=" --served-model-name $SERVED_MODEL_NAME"
    args+=" --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
    args+=" --max-num-batched-tokens $MAX_BATCHED_TOKENS"
    args+=" --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

    if [[ -n "$KV_CACHE_DTYPE" ]]; then
        args+=" --kv-cache-dtype $KV_CACHE_DTYPE"
    fi

    if [[ -n "$MAX_MODEL_LEN" ]]; then
        args+=" --max-model-len $MAX_MODEL_LEN"
    fi

    args+=" --host 0.0.0.0 --port 8000"

    if [[ -n "$EXTRA_ARGS" ]]; then
        args+=" $EXTRA_ARGS"
    fi

    echo "$args"
}

# Export environment variables for Docker Compose
export GPU_DEVICES
export VLLM_PORT="$PORT"

# =============================================================================
# Execute Actions
# =============================================================================

case $ACTION in
    up)
        VLLM_ARGS=$(build_vllm_args)
        export VLLM_ARGS

        echo -e "${GREEN}==============================================================================${NC}"
        echo -e "${GREEN}vLLM Docker Deployment${NC}"
        echo -e "${GREEN}==============================================================================${NC}"
        echo -e "${BLUE}Model:${NC}           $MODEL"
        echo -e "${BLUE}Served Name:${NC}     $SERVED_MODEL_NAME"
        echo -e "${BLUE}GPU Devices:${NC}     $GPU_DEVICES"
        echo -e "${BLUE}Tensor Parallel:${NC} $TENSOR_PARALLEL_SIZE"
        echo -e "${BLUE}Port:${NC}            $PORT"
        echo -e "${BLUE}Max Batch Tokens:${NC} $MAX_BATCHED_TOKENS"
        echo -e "${BLUE}GPU Mem Util:${NC}    $GPU_MEMORY_UTILIZATION"
        if [[ -n "$KV_CACHE_DTYPE" ]]; then
            echo -e "${BLUE}KV Cache Dtype:${NC}  $KV_CACHE_DTYPE"
        fi
        if [[ -n "$MAX_MODEL_LEN" ]]; then
            echo -e "${BLUE}Max Model Len:${NC}   $MAX_MODEL_LEN"
        fi
        if [[ -n "$EXTRA_ARGS" ]]; then
            echo -e "${BLUE}Extra Args:${NC}      $EXTRA_ARGS"
        fi
        echo -e "${GREEN}==============================================================================${NC}"
        echo ""

        # Check if HF_TOKEN is set for gated models
        if [[ "$MODEL" == *"llama"* || "$MODEL" == *"Llama"* ]] && [[ -z "$HF_TOKEN" ]]; then
            echo -e "${YELLOW}Warning: HF_TOKEN not set. LLaMA models require authentication.${NC}"
            echo -e "${YELLOW}Set it with: export HF_TOKEN=your_token${NC}"
        fi

        if [[ "$DETACH" == true ]]; then
            echo -e "${GREEN}Starting vLLM server in background...${NC}"
            docker compose -f "$COMPOSE_FILE" up -d
            echo ""
            echo -e "${GREEN}Server started! View logs with:${NC}"
            echo "  $0 logs"
            echo ""
            echo -e "${GREEN}Test the server:${NC}"
            echo "  curl http://localhost:${PORT}/health"
        else
            echo -e "${GREEN}Starting vLLM server in foreground...${NC}"
            docker compose -f "$COMPOSE_FILE" up
        fi
        ;;

    down)
        echo -e "${YELLOW}Stopping vLLM service${NC}"
        docker compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}Service stopped.${NC}"
        ;;

    logs)
        docker compose -f "$COMPOSE_FILE" logs -f
        ;;

    restart)
        echo -e "${YELLOW}Restarting vLLM service${NC}"
        docker compose -f "$COMPOSE_FILE" restart
        echo -e "${GREEN}Service restarted.${NC}"
        ;;

    ps)
        docker compose -f "$COMPOSE_FILE" ps
        ;;

    *)
        echo -e "${RED}Unknown action: $ACTION${NC}"
        print_usage
        exit 1
        ;;
esac
