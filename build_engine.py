#!/usr/bin/env python3
"""
Production-level TensorRT-LLM Engine Builder
Uses the official TensorRT-LLM LLM API for stable, version-independent builds
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TensorRT-LLM engines using the official LLM API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/models/Meta-Llama-3.1-8B-Instruct --max_seq_len 32768
  %(prog)s ~/models/Qwen3-30B-A3B-Thinking-2507-HF --max_seq_len 32768 --max_num_tokens 8192 --tp 1 --pp 3
  %(prog)s ~/models/Mistral-7B-v0.1 --max_seq_len 16384 --quant awq
        """
    )

    parser.add_argument("model_dir", type=str,
                       help="Path to HuggingFace model directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for engine (default: {model_dir}_engine)")

    # Sequence parameters
    parser.add_argument("--max_seq_len", type=int, default=32768,
                       help="Maximum sequence length (default: 32768)")
    parser.add_argument("--max_num_tokens", type=int, default=8192,
                       help="Maximum number of tokens in batch (default: 8192)")
    parser.add_argument("--max_batch_size", type=int, default=1,
                       help="Maximum batch size (default: 1)")

    # Parallelism
    parser.add_argument("--tp", "--tp_size", type=int, default=1, dest="tp_size",
                       help="Tensor parallelism size (default: 1)")
    parser.add_argument("--pp", "--pp_size", type=int, default=1, dest="pp_size",
                       help="Pipeline parallelism size (default: 1)")

    # Quantization
    parser.add_argument("--quant", type=str, default=None,
                       choices=['awq', 'fp8', 'int8', 'int4'],
                       help="Quantization type (default: none)")

    # Other options
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=['float16', 'bfloat16', 'float32'],
                       help="Data type (default: float16)")
    parser.add_argument("--kv_cache_free_gpu_mem_fraction", type=float, default=0.9,
                       help="KV cache free GPU memory fraction (default: 0.9)")
    parser.add_argument("--clean", action="store_true",
                       help="Clean output directory before building")

    return parser.parse_args()


def get_model_name(model_dir):
    """Extract model name from path"""
    # Handle HuggingFace cache format
    if "snapshots" in model_dir:
        # Go up to the models--Org--Model directory
        parts = Path(model_dir).parts
        for i, part in enumerate(parts):
            if part.startswith("models--"):
                return part
    return Path(model_dir).name


def get_model_class(model_dir):
    """
    Auto-detect model type from HuggingFace config.json and return appropriate TensorRT-LLM class
    Uses TensorRT-LLM's built-in MODEL_MAP for automatic model detection
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    # Log quantization config if present
    quant_config = config.get('quantization_config', {})
    if quant_config:
        quant_method = quant_config.get('quant_method', 'unknown')
        print(f"Note: Model has quantization_config with method '{quant_method}'")

    # Get the model architecture name (e.g., "Qwen3MoeForCausalLM", "LlamaForCausalLM")
    architectures = config.get('architectures', [])
    if not architectures:
        raise ValueError(f"No 'architectures' field found in {config_path}")

    architecture = architectures[0]  # Use first architecture

    # Import TensorRT-LLM's MODEL_MAP
    try:
        from tensorrt_llm.models import MODEL_MAP
    except ImportError as e:
        raise ImportError(f"Failed to import MODEL_MAP from tensorrt_llm.models: {e}")

    # Look up the model class
    if architecture not in MODEL_MAP:
        supported = ', '.join(sorted(set(MODEL_MAP.keys())))
        raise ValueError(
            f"Unsupported model architecture '{architecture}'. "
            f"Supported architectures: {supported}"
        )

    model_class = MODEL_MAP[architecture]
    return model_class, architecture


def main():
    args = parse_args()

    # Resolve paths
    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        output_dir = f"{model_dir}_engine"

    # Clean output directory if requested
    if args.clean and os.path.exists(output_dir):
        print(f"Cleaning old engine directory: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
        print("✓ Cleaned\n")

    model_name = get_model_name(model_dir)

    print("=" * 60)
    print("TensorRT-LLM Engine Builder")
    print("=" * 60)
    print(f"Model:              {model_name}")
    print(f"Source:             {model_dir}")
    print(f"Output:             {output_dir}")
    print(f"Max Seq Len:        {args.max_seq_len}")
    print(f"Max Num Tokens:     {args.max_num_tokens}")
    print(f"Max Batch Size:     {args.max_batch_size}")
    print(f"TP Size:            {args.tp_size}")
    print(f"PP Size:            {args.pp_size}")
    print(f"Data Type:          {args.dtype}")
    print(f"Quantization:       {args.quant or 'none'}")
    print(f"Clean Build:        {args.clean}")
    print("=" * 60)
    print()

    # Detect model type and get appropriate class
    print("Detecting model type...")
    try:
        model_class, model_type = get_model_class(model_dir)
        print(f"✓ Detected: {model_type} -> {model_class.__name__}\n")
    except Exception as e:
        print(f"✗ Error detecting model type: {e}")
        sys.exit(1)

    # Import TensorRT-LLM (do this after argparse so --help works without GPU)
    try:
        import tensorrt_llm
        from tensorrt_llm import BuildConfig
        print(f"Using TensorRT-LLM version: {tensorrt_llm.__version__ if hasattr(tensorrt_llm, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"Error: Failed to import tensorrt_llm: {e}")
        print("Make sure TensorRT-LLM is installed in your environment.")
        sys.exit(1)

    # Calculate world size and prepare output directory
    world_size = args.tp_size * args.pp_size
    os.makedirs(output_dir, exist_ok=True)

    # Create build configuration (shared across all ranks)
    print("Creating build configuration...")
    build_config = BuildConfig(
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
    )
    print(f"✓ Build config created\n")

    # For PP > 1, we MUST use checkpoint conversion (direct API has bugs with PP)
    # For quantization, we also need checkpoint conversion
    if args.quant or args.pp_size > 1:
        # Convert checkpoint first (required for PP>1 and quantization)
        if args.quant:
            print(f"Quantization enabled: {args.quant}")
        if args.pp_size > 1:
            print(f"Pipeline parallelism enabled (PP={args.pp_size})")
        print("Converting checkpoint...")

        ckpt_dir = f"{output_dir}_ckpt"
        if args.clean and os.path.exists(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)

        # Build quantization arguments based on type
        quant_args = {
            'model_dir': model_dir,
            'output_dir': ckpt_dir,
            'dtype': args.dtype,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        }

        if args.quant == 'fp8':
            quant_args['use_weight_only'] = True
            quant_args['weight_only_precision'] = 'fp8'
        elif args.quant == 'int8':
            quant_args['use_weight_only'] = True
            quant_args['weight_only_precision'] = 'int8'
        elif args.quant == 'int4':
            quant_args['use_weight_only'] = True
            quant_args['weight_only_precision'] = 'int4'

        # Convert checkpoint using convert_checkpoint.py script
        # Skip if checkpoint already exists
        if os.path.exists(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "config.json")):
            print(f"✓ Checkpoint already exists at {ckpt_dir}, skipping conversion\n")
        else:
            try:
                import subprocess

                if 'Qwen' in model_type or 'qwen' in model_type.lower():
                    print("Using Qwen checkpoint converter...")

                # Find convert_checkpoint.py
                convert_script = None
                possible_paths = [
                    '/home/steve/src/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py',
                    os.path.expanduser('~/src/TensorRT-LLM/examples/qwen/convert_checkpoint.py'),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        convert_script = path
                        break

                if not convert_script:
                    print(f"✗ Could not find Qwen convert_checkpoint.py")
                    print("Please use the bash script for quantized models")
                    sys.exit(1)

                # Build command
                cmd = [
                    'python', convert_script,
                    '--model_dir', model_dir,
                    '--output_dir', ckpt_dir,
                    '--dtype', args.dtype,
                    '--tp_size', str(args.tp_size),
                    '--pp_size', str(args.pp_size),
                ]

                # Add quantization flags
                if 'use_weight_only' in quant_args:
                    cmd.extend(['--use_weight_only'])
                    cmd.extend(['--weight_only_precision', quant_args['weight_only_precision']])

                print(f"Running: {' '.join(cmd)}\n")
                sys.stdout.flush()
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"\n✗ Checkpoint conversion failed with code {result.returncode}")
                    print("Try running the command manually to see the error:")
                    print(' '.join(cmd))
                    sys.exit(1)

                print(f"✓ Checkpoint converted to {ckpt_dir}\n")

            except Exception as e:
                print(f"✗ Error converting checkpoint: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        # Clean up problematic tensors that RC1 can't handle
        print("Checking for problematic tensors in checkpoint...")
        problematic_tensors = ['embed_positions', 'rotary_inv_freq', 'embed_positions_for_gpt_attention']
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file, save_file
            import glob

            checkpoint_files = glob.glob(os.path.join(ckpt_dir, 'rank*.safetensors'))
            tensors_removed = False

            for ckpt_file in checkpoint_files:
                print(f"  Checking {os.path.basename(ckpt_file)}...", end='', flush=True)
                # Quick check if any problematic tensors exist without loading full file
                needs_cleaning = False
                with safe_open(ckpt_file, framework="pt") as f:
                    for tensor_name in problematic_tensors:
                        if tensor_name in f.keys():
                            needs_cleaning = True
                            break

                if needs_cleaning:
                    print(" needs cleaning")
                    print(f"    Loading checkpoint (this may take a moment)...", flush=True)
                    tensors = load_file(ckpt_file)
                    original_count = len(tensors)

                    # Remove problematic tensors
                    for tensor_name in problematic_tensors:
                        if tensor_name in tensors:
                            del tensors[tensor_name]
                            tensors_removed = True

                    print(f"    Saving cleaned checkpoint...", flush=True)
                    save_file(tensors, ckpt_file)
                    print(f"    ✓ Removed {original_count - len(tensors)} tensors")
                else:
                    print(" OK")

            if tensors_removed:
                print("✓ Removed problematic tensors from checkpoint\n")
            else:
                print("✓ No problematic tensors found\n")

        except Exception as e:
            print(f"⚠ Warning: Could not clean checkpoint tensors: {e}")
            print("  Continuing anyway...\n")

        # Now build from checkpoint instead of HF model
        print(f"Building engine from checkpoint at {ckpt_dir}...")
        import subprocess
        cmd = [
            'trtllm-build',
            '--checkpoint_dir', ckpt_dir,
            '--output_dir', output_dir,
            '--gemm_plugin', 'float16',
            '--max_batch_size', str(args.max_batch_size),
            '--max_input_len', str(args.max_seq_len),
            '--max_seq_len', str(args.max_seq_len),
            '--max_num_tokens', str(args.max_num_tokens),
            '--workers', str(world_size),
        ]

        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"✗ trtllm-build failed with code {result.returncode}")
            sys.exit(1)

    else:
        # No quantization - use direct model loading
        # Build engines for all ranks
        print(f"Building engines for {world_size} ranks (TP={args.tp_size}, PP={args.pp_size})...")
        print(f"This will create: {', '.join([f'rank{r}.engine' for r in range(world_size)])}\n")

        from tensorrt_llm.mapping import Mapping

        for rank in range(world_size):
            print(f"[Rank {rank}/{world_size-1}] Building...")

            # Create mapping for this rank
            mapping = Mapping(
                world_size=world_size,
                rank=rank,
                tp_size=args.tp_size,
                pp_size=args.pp_size
            )

            # Load model from HuggingFace for this rank
            try:
                model = model_class.from_hugging_face(
                    model_dir,
                    dtype=args.dtype,
                    mapping=mapping,
                )
            except Exception as e:
                print(f"✗ Error loading model for rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            # Build engine for this rank
            try:
                engine = tensorrt_llm.build(model, build_config)
            except Exception as e:
                print(f"✗ Error building engine for rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            # Save engine for this rank
            try:
                engine.save(output_dir)
                print(f"✓ Rank {rank} engine saved\n")
            except Exception as e:
                print(f"✗ Error saving engine for rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

    if not args.quant:
        print(f"✓ All {world_size} engines built successfully!\n")
    else:
        print(f"✓ Engine build complete!\n")

    # Copy tokenizer files
    print("Copying tokenizer files to engine directory...")
    import shutil
    tokenizer_files = [
        'generation_config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'tokenizer.model',
        'vocab.json',
        'merges.txt'
    ]

    for fname in tokenizer_files:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {fname}")

    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"\nTest with:")
    print(f"  trtllm-serve serve {output_dir} \\")
    print(f"    --backend tensorrt \\")
    print(f"    --tokenizer {model_dir} \\")
    print(f"    --tp_size {args.tp_size} \\")
    print(f"    --pp_size {args.pp_size} \\")
    print(f"    --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
