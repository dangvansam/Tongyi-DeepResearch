#!/bin/bash

# =============================================================================
# vLLM Model Deployment Script
# Supports: LLaMA, Qwen models with non-quantization, FP8, FP4 options
# =============================================================================

set -e

# Default values
MODEL_TYPE=""
QUANTIZATION="none"  # none, fp8, fp4
TENSOR_PARALLEL_SIZE=1
GPU_DEVICES="0"
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=""
MAX_NUM_BATCHED_TOKENS=8192
ENABLE_MTP=false
KV_CACHE_DTYPE="auto"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -m, --model           Model type: llama70b, llama8b, qwen3-next, qwen3-235b, tongyi-deepresearch"
    echo "  -q, --quantization    Quantization: none, fp8, fp4 (default: none)"
    echo "  -t, --tensor-parallel Tensor parallel size (default: 1)"
    echo "  -g, --gpus            GPU devices (e.g., '0,1,2,3') (default: 0)"
    echo "  -p, --port            Server port (default: 8000)"
    echo "  --host                Server host (default: 0.0.0.0)"
    echo "  --max-model-len       Max model length (optional)"
    echo "  --max-batch-tokens    Max batched tokens (default: 8192)"
    echo "  --enable-mtp          Enable Multi-Token Prediction for Qwen3-Next"
    echo "  --kv-cache-dtype      KV cache dtype: auto, fp8 (default: auto)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  # LLaMA 3.3 70B with FP8 quantization on 2 GPUs"
    echo "  $0 -m llama70b -q fp8 -t 2 -g 0,1"
    echo ""
    echo "  # Qwen3-Next 80B with FP8 on 4 GPUs"
    echo "  $0 -m qwen3-next -q fp8 -t 4 -g 0,1,2,3"
    echo ""
    echo "  # Qwen3-Next 80B with NVFP4"
    echo "  $0 -m qwen3-next -q fp4 -t 4 -g 0,1,2,3"
    echo ""
    echo "  # LLaMA 3.3 70B with FP4 (Blackwell GPUs)"
    echo "  $0 -m llama70b -q fp4 -t 1"
    echo ""
    echo "  # Qwen3-Next with Multi-Token Prediction enabled"
    echo "  $0 -m qwen3-next -t 4 --enable-mtp"
    echo ""
    echo "  # Tongyi-DeepResearch 30B"
    echo "  $0 -m tongyi-deepresearch -t 1 -g 0"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        -t|--tensor-parallel)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_DEVICES="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-batch-tokens)
            MAX_NUM_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --enable-mtp)
            ENABLE_MTP=true
            shift
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate model type
if [[ -z "$MODEL_TYPE" ]]; then
    echo -e "${RED}Error: Model type is required${NC}"
    print_usage
    exit 1
fi

# =============================================================================
# Model Configuration
# =============================================================================

get_model_config() {
    local model_type=$1
    local quant=$2

    case $model_type in
        llama70b|llama-70b|llama3.3-70b)
            case $quant in
                none)
                    MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
                    ;;
                fp8)
                    MODEL_NAME="nvidia/Llama-3.3-70B-Instruct-FP8"
                    ;;
                fp4)
                    MODEL_NAME="nvidia/Llama-3.3-70B-Instruct-FP4"
                    ;;
            esac
            SERVED_NAME="llama-3.3-70b"
            ;;
        llama8b|llama-8b|llama3.1-8b)
            case $quant in
                none)
                    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
                    ;;
                fp8)
                    MODEL_NAME="nvidia/Llama-3.1-8B-Instruct-FP8"
                    ;;
                fp4)
                    echo -e "${YELLOW}Warning: FP4 not officially available for Llama 8B, using FP8${NC}"
                    MODEL_NAME="nvidia/Llama-3.1-8B-Instruct-FP8"
                    ;;
            esac
            SERVED_NAME="llama-3.1-8b"
            ;;
        qwen3-next|qwen3next)
            case $quant in
                none)
                    MODEL_NAME="Qwen/Qwen3-Next-80B-A3B-Instruct"
                    ;;
                fp8)
                    MODEL_NAME="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
                    ;;
                fp4)
                    MODEL_NAME="RESMP-DEV/Qwen3-Next-80B-A3B-Instruct-NVFP4"
                    ;;
            esac
            SERVED_NAME="qwen3-next"
            ;;
        tongyi-deepresearch|deepresearch|tongyi)
            case $quant in
                none|fp8|fp4)
                    MODEL_NAME="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
                    if [[ "$quant" != "none" ]]; then
                        echo -e "${YELLOW}Warning: Tongyi-DeepResearch only available in base precision, using non-quantized${NC}"
                    fi
                    ;;
            esac
            SERVED_NAME="tongyi-deepresearch"
            ;;
        qwen3-235b|qwen3-235)
            case $quant in
                none)
                    MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct"
                    ;;
                fp8)
                    MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-FP8"
                    ;;
                fp4)
                    MODEL_NAME="Qwen/Qwen3-235B-A22B-FP4"
                    ;;
            esac
            SERVED_NAME="qwen3-235b"
            ;;
        qwen3-32b|qwen3-32)
            case $quant in
                none)
                    MODEL_NAME="Qwen/Qwen3-32B"
                    ;;
                fp8)
                    MODEL_NAME="Qwen/Qwen3-32B-FP8"
                    ;;
                fp4)
                    echo -e "${YELLOW}Warning: FP4 not officially available for Qwen3-32B, using FP8${NC}"
                    MODEL_NAME="Qwen/Qwen3-32B-FP8"
                    ;;
            esac
            SERVED_NAME="qwen3-32b"
            ;;
        *)
            echo -e "${RED}Error: Unknown model type: $model_type${NC}"
            echo "Supported models: llama70b, llama8b, qwen3-next, qwen3-235b, qwen3-32b, tongyi-deepresearch"
            exit 1
            ;;
    esac
}

# =============================================================================
# Build vLLM Command
# =============================================================================

build_vllm_command() {
    local cmd="vllm serve $MODEL_NAME"

    # Add tensor parallel size
    cmd+=" --tensor-parallel-size $TENSOR_PARALLEL_SIZE"

    # Add served model name
    cmd+=" --served-model-name $SERVED_NAME"

    # Add host and port
    cmd+=" --host $HOST --port $PORT"

    # Add max model length if specified
    if [[ -n "$MAX_MODEL_LEN" ]]; then
        cmd+=" --max-model-len $MAX_MODEL_LEN"
    fi

    # Add max batched tokens
    cmd+=" --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"

    # Add KV cache dtype
    if [[ "$KV_CACHE_DTYPE" == "fp8" ]]; then
        cmd+=" --kv-cache-dtype fp8"
    fi

    # Model-specific configurations
    case $MODEL_TYPE in
        llama70b|llama-70b|llama3.3-70b)
            # LLaMA specific configs
            cmd+=" --async-scheduling"
            cmd+=" --no-enable-prefix-caching"
            ;;
        qwen3-next|qwen3next)
            # Qwen3-Next specific configs
            if [[ "$ENABLE_MTP" == true ]]; then
                cmd+=" --speculative-config '{\"method\": \"qwen3_next_mtp\", \"num_speculative_tokens\": 2}'"
                cmd+=" --no-enable-chunked-prefill"
                cmd+=" --tokenizer-mode auto"
                cmd+=" --gpu-memory-utilization 0.8"
            fi
            ;;
    esac

    echo "$cmd"
}

# =============================================================================
# Build Environment Variables
# =============================================================================

build_env_vars() {
    local env_vars=""

    # Set GPU devices
    env_vars+="CUDA_VISIBLE_DEVICES=$GPU_DEVICES "

    # Quantization-specific environment variables
    case $QUANTIZATION in
        fp8)
            env_vars+="VLLM_USE_FLASHINFER_MOE_FP8=1 "
            env_vars+="VLLM_FLASHINFER_MOE_BACKEND=latency "
            ;;
        fp4)
            env_vars+="VLLM_USE_FLASHINFER_MOE_FP4=1 "
            env_vars+="VLLM_FLASHINFER_MOE_BACKEND=latency "
            ;;
    esac

    # Additional optimizations for quantized models
    if [[ "$QUANTIZATION" != "none" ]]; then
        env_vars+="VLLM_USE_DEEP_GEMM=0 "
        env_vars+="VLLM_USE_TRTLLM_ATTENTION=0 "
        env_vars+="VLLM_ATTENTION_BACKEND=FLASH_ATTN "
    fi

    echo "$env_vars"
}

# =============================================================================
# Main Execution
# =============================================================================

# Get model configuration
get_model_config "$MODEL_TYPE" "$QUANTIZATION"

# Build environment variables and command
ENV_VARS=$(build_env_vars)
VLLM_CMD=$(build_vllm_command)

# Print configuration
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}vLLM Deployment Configuration${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${BLUE}Model:${NC}           $MODEL_NAME"
echo -e "${BLUE}Served Name:${NC}     $SERVED_NAME"
echo -e "${BLUE}Quantization:${NC}    $QUANTIZATION"
echo -e "${BLUE}Tensor Parallel:${NC} $TENSOR_PARALLEL_SIZE"
echo -e "${BLUE}GPU Devices:${NC}     $GPU_DEVICES"
echo -e "${BLUE}Host:Port:${NC}       $HOST:$PORT"
if [[ "$ENABLE_MTP" == true ]]; then
    echo -e "${BLUE}MTP:${NC}             Enabled"
fi
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo -e "${YELLOW}Environment Variables:${NC}"
echo "$ENV_VARS"
echo ""
echo -e "${YELLOW}Command:${NC}"
echo "$VLLM_CMD"
echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Starting vLLM server...${NC}"
echo -e "${GREEN}==============================================================================${NC}"

# Execute the command
eval "$ENV_VARS $VLLM_CMD"
