#!/usr/bin/env python3
"""
Quantize a HuggingFace model (outputs HuggingFace format)
Supports:
  - int8: SmoothQuant + W8A8 (best INT8 quality)
  - int4: AWQ via autoawq (best INT4 quality)

Output can be used with:
  - vLLM (direct loading)
  - llama.cpp (via convert_hf_to_gguf)
  - Other HuggingFace-compatible tools
"""

import argparse
import os
import sys
from pathlib import Path


def quantize_int8_smoothquant(model_path, output_dir, calibration_samples):
    """Quantize to INT8 using SmoothQuant + W8A8 (best INT8 method)"""
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as e:
        print(f"Error: Failed to import llm-compressor: {e}")
        print("Install with: pip install llmcompressor")
        sys.exit(1)

    print("Creating SmoothQuant + W8A8 recipe (best INT8 method)...")

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(
            scheme="W8A8",
            targets="Linear",
            ignore=["lm_head"],
        ),
    ]

    print("✓ Recipe created\n")
    print("Starting INT8 quantization with SmoothQuant...")
    print("This may take 30-60 minutes for large models...\n")

    oneshot(
        model=model_path,
        dataset="open_platypus",
        output_dir=output_dir,
        num_calibration_samples=calibration_samples,
        recipe=recipe,
        max_seq_length=2048,
    )


def quantize_int4_awq(model_path, output_dir, group_size, calibration_samples):
    """Quantize to INT4 using AWQ (best INT4 method)"""
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Error: Failed to import autoawq: {e}")
        print("Install with: pip install autoawq")
        sys.exit(1)

    print("Loading model for AWQ quantization (best INT4 method)...")
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {
        "w_bit": 4,
        "q_group_size": group_size,
        "zero_point": True,
        "version": "GEMM",
    }

    print(f"Starting AWQ quantization with config: {quant_config}")
    print("This may take 30-60 minutes for large models...\n")

    model.quantize(tokenizer, quant_config=quant_config)

    print("Saving quantized model...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Quantize model for vLLM deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/models/Llama-3.1-8B --bits 4
  %(prog)s ~/models/Qwen3-72B --bits 8 --output ~/models/Qwen3-72B-int8

Quantization methods:
  --bits 8: SmoothQuant + W8A8 (weights AND activations quantized)
            Best INT8 quality, 1.6x speedup at high throughput
  --bits 4: AWQ (activation-aware weight quantization)
            Best INT4 quality, 2x+ memory reduction

Output format: HuggingFace (works with vLLM, convert_hf_to_gguf, etc.)
        """
    )
    parser.add_argument('model_path', type=str, help='Path to FP16 HuggingFace model')
    parser.add_argument('--bits', type=int, required=True, choices=[4, 8],
                       help='Quantization bits (4=AWQ, 8=SmoothQuant W8A8)')
    parser.add_argument('--output', '--output_dir', type=str, default=None, dest='output_dir',
                       help='Output directory (default: {model_path}-int{bits})')
    parser.add_argument('--group-size', type=int, default=128,
                       help='Group size for AWQ (INT4 only, default: 128)')
    parser.add_argument('--calibration-samples', type=int, default=512,
                       help='Number of calibration samples (default: 512)')

    args = parser.parse_args()

    # Resolve paths
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)

    if args.output_dir:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        output_dir = f"{model_path}-int{args.bits}"

    print("=" * 60)
    print(f"INT{args.bits} Quantization for vLLM")
    print("=" * 60)
    print(f"Model:              {model_path}")
    print(f"Output:             {output_dir}")
    print(f"Bits:               {args.bits}")
    if args.bits == 4:
        print(f"Method:             AWQ (best INT4)")
        print(f"Group Size:         {args.group_size}")
    else:
        print(f"Method:             SmoothQuant + W8A8 (best INT8)")
    print(f"Calibration Samples: {args.calibration_samples}")
    print("=" * 60)
    print()

    if args.bits == 8:
        quantize_int8_smoothquant(model_path, output_dir, args.calibration_samples)
    else:
        quantize_int4_awq(model_path, output_dir, args.group_size, args.calibration_samples)

    print("\n✓ Quantization complete\n")

    print("=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"\nQuantized model saved to: {output_dir}")
    print(f"\nUsage:")
    print(f"  vLLM:        vllm serve {output_dir}")
    print(f"  llama.cpp:   python convert_hf_to_gguf.py {output_dir} --outfile model.gguf")
    print()


if __name__ == "__main__":
    main()
