#!/usr/bin/env python3
"""
TensorRT-LLM Engine Builder
Builds optimized TensorRT engines with proper quantization (AWQ, SmoothQuant)
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TensorRT-LLM engines with proper quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/models/Llama-3.1-8B --max_seq_len 32768
  %(prog)s ~/models/Qwen3-72B --quant int4 --tp 1 --pp 3
  %(prog)s ~/models/Mistral-7B --quant int8 --max_seq_len 16384
  %(prog)s ~/models/MyModel-AWQ --tp 1 --pp 3  # auto-detects AWQ

Quantization options:
  int4:  INT4 AWQ (calibrated, best INT4 quality)
  int8:  INT8 SmoothQuant (calibrated, best INT8 quality)
  fp8:   FP8 (Hopper/Ada GPUs only)
  none:  FP16 (no quantization)
        """
    )

    parser.add_argument("model_dir", type=str,
                       help="Path to HuggingFace model directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for engine (default: {model_dir}_engine)")

    # Sequence parameters
    parser.add_argument("--max_seq_len", type=int, default=32768,
                       help="Maximum sequence length (default: 32768)")
    parser.add_argument("--max_num_tokens", type=int, default=None,
                       help="Maximum tokens in batch (default: max_seq_len * max_batch_size)")
    parser.add_argument("--max_batch_size", type=int, default=1,
                       help="Maximum batch size (default: 1)")

    # Parallelism
    parser.add_argument("--tp", "--tp_size", type=int, default=1, dest="tp_size",
                       help="Tensor parallelism size (default: 1)")
    parser.add_argument("--pp", "--pp_size", type=int, default=1, dest="pp_size",
                       help="Pipeline parallelism size (default: 1)")

    # Quantization
    parser.add_argument("--quant", type=str, default=None,
                       choices=['int4', 'int8', 'fp8'],
                       help="Quantization: int4=AWQ, int8=SmoothQuant, fp8=FP8")
    parser.add_argument("--calib_size", type=int, default=512,
                       help="Calibration samples for quantization (default: 512)")
    parser.add_argument("--awq_block_size", type=int, default=128,
                       help="AWQ block size (default: 128)")

    # Other options
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=['float16', 'bfloat16'],
                       help="Base data type (default: float16)")
    parser.add_argument("--clean", action="store_true",
                       help="Clean output directory before building")
    parser.add_argument("--workers", type=int, default=None,
                       help="Parallel workers for engine build (default: 1 for safety)")
    parser.add_argument("--quantize-only", action="store_true",
                       help="Only create quantized checkpoint, don't build engine (for running on high-RAM CPU-only systems)")
    parser.add_argument("--build-only", action="store_true",
                       help="Only build engine from existing checkpoint (for running on GPU systems)")
    parser.add_argument("--cpu", action="store_true",
                       help="Run quantization on CPU (slower but uses system RAM instead of GPU memory)")

    return parser.parse_args()


def get_model_name(model_dir):
    """Extract model name from path"""
    if "snapshots" in model_dir:
        parts = Path(model_dir).parts
        for part in parts:
            if part.startswith("models--"):
                return part
    return Path(model_dir).name


def quantize_checkpoint(model_dir, output_dir, qformat, tp_size, pp_size,
                        dtype, calib_size, awq_block_size, use_cpu=False):
    """Quantize model using TensorRT-LLM's quantize_and_export API"""
    import logging
    from tensorrt_llm.quantization import quantize_and_export
    from tensorrt_llm.logger import logger as trtllm_logger

    # Enable DEBUG logging to see per-batch calibration progress
    trtllm_logger.set_level('verbose')
    logging.getLogger("tensorrt_llm").setLevel(logging.DEBUG)

    device = 'cpu' if use_cpu else 'cuda'
    device_map = 'cpu' if use_cpu else 'auto'

    print(f"Quantizing with qformat={qformat}...")
    print(f"  Device: {device}")
    print(f"  Calibration samples: {calib_size}")
    if 'awq' in qformat:
        print(f"  AWQ block size: {awq_block_size}")
    print()
    sys.stdout.flush()

    quantize_and_export(
        model_dir=model_dir,
        output_dir=output_dir,
        dtype=dtype,
        qformat=qformat,
        calib_dataset='cnn_dailymail',
        calib_size=calib_size,
        calib_max_seq_length=2048,
        batch_size=1,
        awq_block_size=awq_block_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=1,
        seed=42,
        device=device,
        device_map=device_map,
        kv_cache_dtype=None,
        tokenizer_max_seq_length=4096,
    )

    print(f"✓ Quantized checkpoint saved to {output_dir}\n")


def convert_checkpoint(model_dir, output_dir, tp_size, pp_size, dtype, quant_type=None):
    """Convert HuggingFace checkpoint to TensorRT-LLM format using Python API

    Args:
        quant_type: 'awq', 'gptq', or None for unquantized models
    """
    from tensorrt_llm.models import MODEL_MAP
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo
    from tensorrt_llm.mapping import Mapping

    # Detect model architecture
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    architectures = config.get('architectures', [])
    if not architectures:
        raise ValueError(f"No 'architectures' field found in {config_path}")

    arch = architectures[0]
    if arch not in MODEL_MAP:
        raise ValueError(f"Unsupported architecture '{arch}'")

    model_class = MODEL_MAP[arch]
    hf_quant_config = config.get('quantization_config', {})

    if quant_type == 'awq':
        print(f"Converting pre-quantized AWQ model...")
        print(f"  Architecture: {arch} -> {model_class.__name__}")
        print(f"  Bits: {hf_quant_config.get('bits', 4)}")
        print(f"  Group size: {hf_quant_config.get('group_size', 128)}")
    elif quant_type == 'gptq':
        print(f"Converting pre-quantized GPTQ model...")
        print(f"  Architecture: {arch} -> {model_class.__name__}")
        print(f"  Bits: {hf_quant_config.get('bits', 4)}")
        print(f"  Group size: {hf_quant_config.get('group_size', 128)}")
    else:
        print(f"Converting checkpoint...")
        print(f"  Architecture: {arch} -> {model_class.__name__}")

    world_size = tp_size * pp_size

    # Set up quantization config for pre-quantized models
    tllm_quant_config = None
    override_fields = {}

    if quant_type == 'awq':
        # AutoAWQ uses W4A16_GPTQ algorithm (asymmetric UINT4 with zero points)
        # but needs use_autoawq=True for correct tensor loading (different format than GPTQ)
        tllm_quant_config = QuantConfig()
        tllm_quant_config.quant_algo = QuantAlgo.W4A16_GPTQ
        tllm_quant_config.group_size = hf_quant_config.get('group_size', 128)
        tllm_quant_config.has_zero_point = hf_quant_config.get('zero_point', True)
        override_fields['use_autoawq'] = True
    elif quant_type == 'gptq':
        tllm_quant_config = QuantConfig()
        tllm_quant_config.quant_algo = QuantAlgo.W4A16_GPTQ
        tllm_quant_config.group_size = hf_quant_config.get('group_size', 128)
        tllm_quant_config.has_zero_point = not hf_quant_config.get('sym', True)

    # Convert for each rank
    for rank in range(world_size):
        print(f"  Converting rank {rank+1}/{world_size}...")

        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=tp_size,
            pp_size=pp_size
        )

        # Load model
        load_kwargs = {
            'dtype': dtype,
            'mapping': mapping,
        }
        if tllm_quant_config:
            load_kwargs['quant_config'] = tllm_quant_config
        load_kwargs.update(override_fields)

        model = model_class.from_hugging_face(model_dir, **load_kwargs)

        model.save_checkpoint(output_dir, save_config=(rank == 0))

    print(f"✓ Checkpoint converted to {output_dir}\n")


def clean_checkpoint_tensors(ckpt_dir):
    """Remove problematic tensors that cause TRT-LLM build failures"""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import glob

    problematic_tensors = ['embed_positions', 'rotary_inv_freq', 'embed_positions_for_gpt_attention']
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, 'rank*.safetensors'))

    print("Checking for problematic tensors...")
    for ckpt_file in checkpoint_files:
        print(f"  {os.path.basename(ckpt_file)}...", end='', flush=True)

        needs_cleaning = False
        with safe_open(ckpt_file, framework="pt") as f:
            for tensor_name in problematic_tensors:
                if tensor_name in f.keys():
                    needs_cleaning = True
                    break

        if needs_cleaning:
            print(" cleaning...", end='', flush=True)
            tensors = load_file(ckpt_file)
            original_count = len(tensors)
            for tensor_name in problematic_tensors:
                if tensor_name in tensors:
                    del tensors[tensor_name]
            save_file(tensors, ckpt_file)
            print(f" removed {original_count - len(tensors)} tensors")
        else:
            print(" OK")
    print()


def fix_scales_for_trtllm_build(ckpt_dir):
    """Transpose scales so trtllm-build preprocessing produces correct shape.

    TRT-LLM's from_hugging_face saves scales as [groups, out].
    But trtllm-build's preprocess_weights transposes them AGAIN.
    So we pre-transpose here to [out, groups] so final result is [groups, out].
    """
    import glob
    from safetensors.torch import load_file, save_file

    checkpoint_files = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
    if not checkpoint_files:
        return

    print("Fixing scales for trtllm-build...")
    for ckpt_file in checkpoint_files:
        print(f"  {os.path.basename(ckpt_file)}...", end='', flush=True)
        tensors = load_file(ckpt_file)
        modified = False

        for name in list(tensors.keys()):
            if 'weights_scaling_factor' in name:
                t = tensors[name]
                # Check if needs transpose (should be [groups, out], preprocess expects [out, groups])
                # Transpose so preprocess produces [groups, out]
                tensors[name] = t.t().contiguous()
                modified = True

        if modified:
            save_file(tensors, ckpt_file)
            print(" transposed scales")
        else:
            print(" OK")
    print()


def fix_checkpoint_version(ckpt_dir):
    """Fix checkpoint config for version compatibility"""
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        config = json.load(f)

    modified = False

    # Remove fields that don't exist in older TRT-LLM versions
    if 'mapping' in config:
        if 'enable_lm_head_tp_in_adp' in config['mapping']:
            del config['mapping']['enable_lm_head_tp_in_adp']
            modified = True

    if modified:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Fixed checkpoint config for TRT-LLM version compatibility\n")


def build_engine(ckpt_dir, output_dir, max_batch_size, max_seq_len, max_num_tokens, workers):
    """Build TensorRT engine from checkpoint"""
    import subprocess

    fix_checkpoint_version(ckpt_dir)

    cmd = [
        'trtllm-build',
        '--checkpoint_dir', ckpt_dir,
        '--output_dir', output_dir,
        '--gemm_plugin', 'float16',
        '--max_batch_size', str(max_batch_size),
        '--max_input_len', str(max_seq_len),
        '--max_seq_len', str(max_seq_len),
        '--max_num_tokens', str(max_num_tokens),
        '--workers', str(workers),
    ]

    print(f"Building TensorRT engine...")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"trtllm-build failed with code {result.returncode}")

    print(f"✓ Engine built successfully\n")


def copy_tokenizer_files(model_dir, output_dir):
    """Copy tokenizer files to engine directory"""
    tokenizer_files = [
        'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'tokenizer.model', 'vocab.json',
        'merges.txt', 'chat_template.jinja'
    ]

    print("Copying tokenizer files...")
    for fname in tokenizer_files:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  ✓ {fname}")
    print()


def main():
    args = parse_args()

    # Resolve paths
    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    output_dir = args.output_dir or f"{model_dir}_engine"
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    ckpt_dir = f"{output_dir}_ckpt"

    # Clean if requested
    if args.clean:
        for d in [output_dir, ckpt_dir]:
            if os.path.exists(d):
                print(f"Cleaning {d}...")
                shutil.rmtree(d)

    # Map quant options to TRT-LLM qformat
    qformat_map = {
        'int4': 'int4_awq',      # Best INT4
        'int8': 'int8_sq',       # Best INT8 (SmoothQuant)
        'fp8': 'fp8',            # FP8 (Hopper/Ada)
    }
    qformat = qformat_map.get(args.quant) if args.quant else None

    world_size = args.tp_size * args.pp_size
    workers = args.workers or 1  # Default to 1 for memory safety
    max_num_tokens = args.max_num_tokens or (args.max_seq_len * args.max_batch_size)

    # Print configuration
    model_name = get_model_name(model_dir)
    print("=" * 60)
    print("TensorRT-LLM Engine Builder")
    print("=" * 60)
    print(f"Model:           {model_name}")
    print(f"Source:          {model_dir}")
    print(f"Output:          {output_dir}")
    print(f"Max Seq Len:     {args.max_seq_len}")
    print(f"Max Num Tokens:  {max_num_tokens}")
    print(f"Max Batch Size:  {args.max_batch_size}")
    print(f"TP Size:         {args.tp_size}")
    print(f"PP Size:         {args.pp_size}")
    # Check if model is pre-quantized
    model_config_path = os.path.join(model_dir, "config.json")
    is_prequantized_awq = False
    is_prequantized_gptq = False
    unsupported_quant_format = None
    if os.path.exists(model_config_path):
        with open(model_config_path) as f:
            model_config = json.load(f)
        quant_cfg = model_config.get('quantization_config', {})
        quant_method = quant_cfg.get('quant_method', '').lower()

        if quant_method == 'awq':
            is_prequantized_awq = True
        elif quant_method == 'gptq':
            is_prequantized_gptq = True
        elif quant_method in ['compressed-tensors', 'compressed_tensors']:
            # compressed-tensors is llmcompressor/vLLM format, NOT compatible with TRT-LLM
            unsupported_quant_format = 'compressed-tensors'
        elif quant_method:
            unsupported_quant_format = quant_method

    if unsupported_quant_format:
        print(f"Quantization:    {unsupported_quant_format} (UNSUPPORTED by TRT-LLM!)")
        print()
        print("=" * 60)
        print("ERROR: Unsupported quantization format!")
        print("=" * 60)
        print(f"This model uses '{unsupported_quant_format}' quantization format.")
        print()
        if unsupported_quant_format == 'compressed-tensors':
            print("The 'compressed-tensors' format is used by llmcompressor/vLLM")
            print("but is NOT compatible with TensorRT-LLM.")
            print()
            print("TensorRT-LLM only supports:")
            print("  - 'awq' (AutoAWQ format)")
            print("  - 'gptq' (GPTQ format)")
            print()
            print("Options:")
            print("  1. Find the same model quantized with AutoAWQ instead")
            print("  2. Re-quantize the base model using modelopt/TRT-LLM's quantize.py")
            print("  3. Use vLLM instead of TRT-LLM for this model")
        sys.exit(1)
    elif is_prequantized_awq:
        print(f"Quantization:    pre-quantized AWQ (auto-detected, no calibration)")
    elif is_prequantized_gptq:
        print(f"Quantization:    pre-quantized GPTQ (auto-detected, no calibration)")
    else:
        print(f"Quantization:    {args.quant or 'none'} ({qformat or 'fp16'})")
        if qformat:
            print(f"Calib Samples:   {args.calib_size}")
    print(f"Build Workers:   {workers}")
    print("=" * 60)
    print()

    # Import TensorRT-LLM
    try:
        import tensorrt_llm
        print(f"TensorRT-LLM version: {getattr(tensorrt_llm, '__version__', 'unknown')}\n")
    except ImportError as e:
        print(f"Error: Failed to import tensorrt_llm: {e}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Step 1: Create quantized checkpoint (skip if --build-only)
    ckpt_exists = os.path.exists(os.path.join(ckpt_dir, "config.json"))

    if args.build_only and not ckpt_exists:
        print(f"Error: --build-only specified but no checkpoint found at {ckpt_dir}")
        sys.exit(1)

    if args.build_only:
        print(f"✓ Using existing checkpoint at {ckpt_dir} (--build-only)\n")
    elif not ckpt_exists:
        if is_prequantized_awq:
            # Convert pre-quantized AWQ model (no calibration needed)
            convert_checkpoint(
                model_dir=model_dir,
                output_dir=ckpt_dir,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                dtype=args.dtype,
                quant_type='awq',
            )
        elif is_prequantized_gptq:
            # Convert pre-quantized GPTQ model (no calibration needed)
            convert_checkpoint(
                model_dir=model_dir,
                output_dir=ckpt_dir,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                dtype=args.dtype,
                quant_type='gptq',
            )
        elif qformat:
            # Use quantize_and_export for calibrated quantization
            quantize_checkpoint(
                model_dir=model_dir,
                output_dir=ckpt_dir,
                qformat=qformat,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                dtype=args.dtype,
                calib_size=args.calib_size,
                awq_block_size=args.awq_block_size,
                use_cpu=args.cpu,
            )
        else:
            # No quant - just convert checkpoint
            convert_checkpoint(
                model_dir=model_dir,
                output_dir=ckpt_dir,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                dtype=args.dtype,
            )
    else:
        print(f"✓ Using existing checkpoint at {ckpt_dir}\n")

    # Step 2: Clean problematic tensors
    try:
        clean_checkpoint_tensors(ckpt_dir)
    except Exception as e:
        print(f"Warning: Could not clean tensors: {e}\n")

    # Stop here if --quantize-only
    if args.quantize_only:
        print("=" * 60)
        print("Quantization Complete! (--quantize-only)")
        print("=" * 60)
        print(f"\nCheckpoint saved to: {ckpt_dir}")
        print(f"\nTo build engine on GPU system:")
        print(f"  ./build_engine.py {model_dir} --quant {args.quant} --tp {args.tp_size} --pp {args.pp_size} --build-only")
        print()
        return

    # Step 3: Build engine
    build_engine(
        ckpt_dir=ckpt_dir,
        output_dir=output_dir,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=max_num_tokens,
        workers=workers,
    )

    # Step 4: Copy tokenizer
    copy_tokenizer_files(model_dir, output_dir)

    # Done
    print("=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"\nEngine saved to: {output_dir}")
    print(f"\nServe with:")
    print(f"  shepherd --server --backend tensorrt --model {output_dir} --tp {args.tp_size} --pp {args.pp_size}")
    print()


if __name__ == "__main__":
    main()
