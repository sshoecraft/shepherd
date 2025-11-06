#!/bin/bash
set -e

# Unified TensorRT-LLM Engine Build Script
# Usage: ./build_engine.sh <model_dir> <max_seq_len> [max_batch_size] [--clean] [--tp=N] [--pp=N]
# Example: ./build_engine.sh ~/models/Meta-Llama-3.1-8B-Instruct 32768 4 --clean --tp=1 --pp=2

VENV=/home/steve/venv

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_dir> <max_seq_len> [max_batch_size] [--clean] [--tp=N] [--pp=N]"
    echo ""
    echo "Arguments:"
    echo "  model_dir       Path to HuggingFace model directory"
    echo "  max_seq_len     Maximum sequence length (e.g., 4096, 32768)"
    echo "  max_batch_size  Maximum batch size (default: 4)"
    echo "  --clean         Clean old checkpoint and engine directories before building"
    echo "  --tp=N          Tensor parallelism size (default: 2)"
    echo "  --pp=N          Pipeline parallelism size (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0 ~/models/Meta-Llama-3.1-8B-Instruct 32768 4 --clean"
    echo "  $0 ~/models/Meta-Llama-3.1-8B-Instruct 32768 4 --tp=1 --pp=2"
    echo "  $0 ~/models/Llama-3.1-8B-Instruct-FP8 4096 8 --clean --tp=2 --pp=1"
    exit 1
fi

MODEL_DIR="$1"
MAX_SEQ_LEN="$2"
MAX_BATCH_SIZE="${3:-4}"  # Default to 4 if not specified
CLEAN=false
TP_SIZE=2  # Default to 2
PP_SIZE=1  # Default to 1

# Check for flags
for arg in "$@"; do
    if [ "$arg" = "--clean" ]; then
        CLEAN=true
    elif [[ "$arg" == --tp=* ]]; then
        TP_SIZE="${arg#--tp=}"
    elif [[ "$arg" == --pp=* ]]; then
        PP_SIZE="${arg#--pp=}"
    fi
done

# Validate model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Extract model name from path for output directories
MODEL_NAME=$(basename "$MODEL_DIR")
CKPT_DIR="${MODEL_DIR}_ckpt"
ENGINE_DIR="${MODEL_DIR}_engine"
LOG_FILE="${MODEL_DIR}_build.log"

# Detect model type from config.json before logging
MODEL_TYPE="llama"  # Default to llama
DETECTED_TYPE="unknown"
if [ -f "$MODEL_DIR/config.json" ]; then
    # Extract model_type from config.json
    DETECTED_TYPE=$($VENV/bin/python -c "import json; f=open('$MODEL_DIR/config.json'); c=json.load(f); print(c.get('model_type', 'llama'))" 2>/dev/null || echo "llama")

    if [[ "$DETECTED_TYPE" == qwen* ]]; then
        MODEL_TYPE="qwen"
    else
        MODEL_TYPE="llama"
    fi
fi

# Redirect all output to log file and console
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "TensorRT-LLM Engine Build Configuration"
echo "=========================================="
echo "Model:          $MODEL_NAME"
echo "Model Type:     $DETECTED_TYPE"
echo "Converter:      $MODEL_TYPE"
echo "Source:         $MODEL_DIR"
echo "Checkpoint:     $CKPT_DIR"
echo "Engine:         $ENGINE_DIR"
echo "Log File:       $LOG_FILE"
echo "Max Seq Len:    $MAX_SEQ_LEN"
echo "Max Batch Size: $MAX_BATCH_SIZE"
echo "TP Size:        $TP_SIZE"
echo "PP Size:        $PP_SIZE"
echo "Data Type:      float16"
echo "Clean Build:    $CLEAN"
echo "=========================================="
echo ""

# Clean old builds if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning old checkpoint and engine directories..."
    rm -rf "$CKPT_DIR" "$ENGINE_DIR"
fi

# Step 1: Convert checkpoint
if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR 2>/dev/null)" ]; then
    echo "Step 1: Checkpoint already exists at $CKPT_DIR, skipping conversion..."
else
    echo "Step 1: Converting checkpoint..."
    source $VENV/bin/activate

    if [ "$MODEL_TYPE" = "qwen" ]; then
        # Use Qwen converter (supports MoE)
        echo "Using Qwen converter..."
        python /home/steve/src/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py \
            --model_dir "$MODEL_DIR" \
            --output_dir "$CKPT_DIR" \
            --dtype float16 \
            --tp_size $TP_SIZE \
            --pp_size $PP_SIZE \
            --moe_tp_size $TP_SIZE \
            --moe_ep_size 1
    else
        # Use LLaMA converter
        echo "Using LLaMA converter..."
        python /home/steve/src/TensorRT-LLM/examples/models/core/llama/convert_checkpoint.py \
            --model_dir "$MODEL_DIR" \
            --output_dir "$CKPT_DIR" \
            --dtype float16 \
            --tp_size $TP_SIZE \
            --pp_size $PP_SIZE
    fi
fi

# Step 2: Build TensorRT engine
echo ""
echo "Step 2: Building TensorRT engine..."
MAX_INPUT_LEN=$MAX_SEQ_LEN

# Determine number of workers based on TP/PP size
WORKERS=$((TP_SIZE * PP_SIZE))

~/venv/bin/python ~/venv/bin/trtllm-build \
    --checkpoint_dir "$CKPT_DIR" \
    --output_dir "$ENGINE_DIR" \
    --gemm_plugin float16 \
    --max_batch_size $MAX_BATCH_SIZE \
    --max_input_len $MAX_INPUT_LEN \
    --max_seq_len $MAX_SEQ_LEN \
    --max_num_tokens 2048 \
    --workers $WORKERS

# Step 3: Copy tokenizer files to engine directory
echo ""
echo "Step 3: Copying tokenizer files to engine directory..."
for tokenizer_file in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
    if [ -f "$MODEL_DIR/$tokenizer_file" ]; then
        cp "$MODEL_DIR/$tokenizer_file" "$ENGINE_DIR/"
        echo "Copied $tokenizer_file to engine directory"
    fi
done

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Engine location: $ENGINE_DIR"
echo ""
echo "Test with shepherd:"
echo "  ./shepherd --backend tensorrt --model $ENGINE_DIR"
echo ""
echo "Or test with TensorRT-LLM directly:"
echo "  mpirun -n 2 python ~/src/TensorRT-LLM/examples/run.py \\"
echo "    --engine_dir $ENGINE_DIR \\"
echo "    --tokenizer_dir $MODEL_DIR \\"
echo "    --max_output_len 50 \\"
echo "    --input_text 'Hello, how are you?'"
