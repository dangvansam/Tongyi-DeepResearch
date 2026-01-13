#!/bin/bash

# =============================================================================
# vLLM Benchmark Script
# Benchmarks vLLM server performance after deployment
# =============================================================================

set -e

# Default values
MODEL_NAME=""
SERVED_MODEL_NAME=""
HOST="0.0.0.0"
PORT=8000
ENDPOINT="/v1/completions"
DATASET_NAME="random"
RANDOM_INPUT_LEN=2048
RANDOM_OUTPUT_LEN=1024
MAX_CONCURRENCY=10
NUM_PROMPTS=100
TRUST_REMOTE_CODE=true
IGNORE_EOS=true
SAVE_RESULT=false
RESULT_FILENAME="vllm_benchmark_results.json"

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
    echo "  -m, --model           Model name on HuggingFace (required)"
    echo "  -s, --served-name     Served model name (default: derived from model)"
    echo "  --host                Server host (default: 0.0.0.0)"
    echo "  -p, --port            Server port (default: 8000)"
    echo "  --endpoint            API endpoint (default: /v1/completions)"
    echo "  --dataset             Dataset name: random, sharegpt, sonnet (default: random)"
    echo "  --input-len           Random input length (default: 2048)"
    echo "  --output-len          Random output length (default: 1024)"
    echo "  --concurrency         Max concurrency (default: 10)"
    echo "  --num-prompts         Number of prompts (default: 100)"
    echo "  --save                Save results to JSON file"
    echo "  --result-file         Result filename (default: vllm_benchmark_results.json)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo -e "${BLUE}Preset Models (use -m with these shortcuts):${NC}"
    echo "  llama70b              meta-llama/Llama-3.3-70B-Instruct"
    echo "  llama70b-fp8          nvidia/Llama-3.3-70B-Instruct-FP8"
    echo "  llama70b-fp4          nvidia/Llama-3.3-70B-Instruct-FP4"
    echo "  qwen3-next            Qwen/Qwen3-Next-80B-A3B-Instruct"
    echo "  qwen3-next-fp8        Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    echo "  qwen3-next-fp4        RESMP-DEV/Qwen3-Next-80B-A3B-Instruct-NVFP4"
    echo "  tongyi-deepresearch   Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  # Basic benchmark with Qwen3-Next"
    echo "  $0 -m qwen3-next"
    echo ""
    echo "  # Benchmark with custom parameters"
    echo "  $0 -m qwen3-next --input-len 1024 --output-len 512 --concurrency 20 --num-prompts 200"
    echo ""
    echo "  # Benchmark Tongyi-DeepResearch and save results"
    echo "  $0 -m tongyi-deepresearch --save --result-file deepresearch_bench.json"
    echo ""
    echo "  # Benchmark with custom model path"
    echo "  $0 -m Qwen/Qwen3-32B -s qwen3-32b"
}

# Resolve model shortcuts to full paths
resolve_model_name() {
    local input=$1
    case $input in
        llama70b)
            MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
            SERVED_MODEL_NAME="llama-3.3-70b"
            ;;
        llama70b-fp8)
            MODEL_NAME="nvidia/Llama-3.3-70B-Instruct-FP8"
            SERVED_MODEL_NAME="llama-3.3-70b"
            ;;
        llama70b-fp4)
            MODEL_NAME="nvidia/Llama-3.3-70B-Instruct-FP4"
            SERVED_MODEL_NAME="llama-3.3-70b"
            ;;
        llama8b)
            MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
            SERVED_MODEL_NAME="llama-3.1-8b"
            ;;
        llama8b-fp8)
            MODEL_NAME="nvidia/Llama-3.1-8B-Instruct-FP8"
            SERVED_MODEL_NAME="llama-3.1-8b"
            ;;
        qwen3-next)
            MODEL_NAME="Qwen/Qwen3-Next-80B-A3B-Instruct"
            SERVED_MODEL_NAME="qwen3-next"
            ;;
        qwen3-next-fp8)
            MODEL_NAME="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
            SERVED_MODEL_NAME="qwen3-next"
            ;;
        qwen3-next-fp4)
            MODEL_NAME="RESMP-DEV/Qwen3-Next-80B-A3B-Instruct-NVFP4"
            SERVED_MODEL_NAME="qwen3-next"
            ;;
        tongyi-deepresearch|deepresearch|tongyi)
            MODEL_NAME="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
            SERVED_MODEL_NAME="tongyi-deepresearch"
            ;;
        qwen3-235b)
            MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct"
            SERVED_MODEL_NAME="qwen3-235b"
            ;;
        qwen3-32b)
            MODEL_NAME="Qwen/Qwen3-32B"
            SERVED_MODEL_NAME="qwen3-32b"
            ;;
        *)
            # Assume it's a full model path
            MODEL_NAME="$input"
            # Derive served name from model path if not set
            if [[ -z "$SERVED_MODEL_NAME" ]]; then
                SERVED_MODEL_NAME=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
            fi
            ;;
    esac
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_INPUT="$2"
            shift 2
            ;;
        -s|--served-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --input-len)
            RANDOM_INPUT_LEN="$2"
            shift 2
            ;;
        --output-len)
            RANDOM_OUTPUT_LEN="$2"
            shift 2
            ;;
        --concurrency)
            MAX_CONCURRENCY="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --save)
            SAVE_RESULT=true
            shift
            ;;
        --result-file)
            RESULT_FILENAME="$2"
            SAVE_RESULT=true
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

# Validate model
if [[ -z "$MODEL_INPUT" ]]; then
    echo -e "${RED}Error: Model name is required${NC}"
    print_usage
    exit 1
fi

# Resolve model name
resolve_model_name "$MODEL_INPUT"

# =============================================================================
# Check Server Health
# =============================================================================

check_server_health() {
    echo -e "${YELLOW}Checking vLLM server health...${NC}"

    local max_retries=5
    local retry_count=0
    local server_url="http://${HOST}:${PORT}/health"

    while [[ $retry_count -lt $max_retries ]]; do
        if curl -s --connect-timeout 5 "$server_url" > /dev/null 2>&1; then
            echo -e "${GREEN}Server is healthy!${NC}"
            return 0
        fi

        retry_count=$((retry_count + 1))
        echo -e "${YELLOW}Waiting for server... (attempt $retry_count/$max_retries)${NC}"
        sleep 5
    done

    echo -e "${RED}Error: Server at ${HOST}:${PORT} is not responding${NC}"
    echo "Make sure the vLLM server is running with:"
    echo "  ./deploy_vllm.sh -m <model> ..."
    exit 1
}

# =============================================================================
# Build Benchmark Command
# =============================================================================

build_benchmark_command() {
    local cmd="vllm bench serve"

    cmd+=" --backend vllm"
    cmd+=" --model $MODEL_NAME"
    cmd+=" --served-model-name $SERVED_MODEL_NAME"
    cmd+=" --host $HOST"
    cmd+=" --port $PORT"
    cmd+=" --endpoint $ENDPOINT"
    cmd+=" --dataset-name $DATASET_NAME"

    # Dataset-specific options
    if [[ "$DATASET_NAME" == "random" ]]; then
        cmd+=" --random-input-len $RANDOM_INPUT_LEN"
        cmd+=" --random-output-len $RANDOM_OUTPUT_LEN"
    fi

    cmd+=" --max-concurrency $MAX_CONCURRENCY"
    cmd+=" --num-prompts $NUM_PROMPTS"

    if [[ "$TRUST_REMOTE_CODE" == true ]]; then
        cmd+=" --trust-remote-code"
    fi

    if [[ "$IGNORE_EOS" == true ]]; then
        cmd+=" --ignore-eos"
    fi

    if [[ "$SAVE_RESULT" == true ]]; then
        cmd+=" --save-result --result-filename $RESULT_FILENAME"
    fi

    echo "$cmd"
}

# =============================================================================
# Main Execution
# =============================================================================

# Check server health first
check_server_health

# Build command
BENCH_CMD=$(build_benchmark_command)

# Print configuration
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}vLLM Benchmark Configuration${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${BLUE}Model:${NC}           $MODEL_NAME"
echo -e "${BLUE}Served Name:${NC}     $SERVED_MODEL_NAME"
echo -e "${BLUE}Server:${NC}          http://${HOST}:${PORT}"
echo -e "${BLUE}Endpoint:${NC}        $ENDPOINT"
echo -e "${BLUE}Dataset:${NC}         $DATASET_NAME"
if [[ "$DATASET_NAME" == "random" ]]; then
    echo -e "${BLUE}Input Length:${NC}    $RANDOM_INPUT_LEN"
    echo -e "${BLUE}Output Length:${NC}   $RANDOM_OUTPUT_LEN"
fi
echo -e "${BLUE}Concurrency:${NC}     $MAX_CONCURRENCY"
echo -e "${BLUE}Num Prompts:${NC}     $NUM_PROMPTS"
if [[ "$SAVE_RESULT" == true ]]; then
    echo -e "${BLUE}Result File:${NC}     $RESULT_FILENAME"
fi
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo -e "${YELLOW}Command:${NC}"
echo "$BENCH_CMD"
echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Starting benchmark...${NC}"
echo -e "${GREEN}==============================================================================${NC}"

# Execute benchmark
eval "$BENCH_CMD"

# Print completion message
echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Benchmark completed!${NC}"
if [[ "$SAVE_RESULT" == true ]]; then
    echo -e "${GREEN}Results saved to: ${RESULT_FILENAME}${NC}"
fi
echo -e "${GREEN}==============================================================================${NC}"
