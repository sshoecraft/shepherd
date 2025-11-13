#!/usr/bin/env python3
"""
Quantize a HuggingFace model to various formats
Uses llm-compressor for weight-only quantization
Supports: INT8, INT4 (AWQ), FP8
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Quantize model using llm-compressor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/models/Llama-3.1-8B-Instruct --bits 4
  %(prog)s ~/models/Qwen3-30B --bits 8 --output ~/models/Qwen3-30B-int8
  %(prog)s ~/models/Mistral-7B --bits 4 --group-size 64
        """
    )
    parser.add_argument('model_path', type=str, help='Path to FP16 HuggingFace model')
    parser.add_argument('--bits', type=int, required=True, choices=[4, 8],
                       help='Quantization bits (4=AWQ, 8=INT8)')
    parser.add_argument('--output', '--output_dir', type=str, default=None, dest='output_dir',
                       help='Output directory (default: {model_path}_int8 or _awq)')
    parser.add_argument('--group-size', type=int, default=128,
                       help='Group size for quantization (default: 128)')
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
        suffix = 'awq' if args.bits == 4 else 'int8'
        output_dir = f"{model_path}_{suffix}"

    # Determine quantization type
    quant_name = 'AWQ (4-bit)' if args.bits == 4 else 'INT8'
    symmetric = args.bits == 8  # INT8 uses symmetric, AWQ uses asymmetric

    print("=" * 60)
    print(f"{quant_name} Quantization (using llm-compressor)")
    print("=" * 60)
    print(f"Model:              {model_path}")
    print(f"Output:             {output_dir}")
    print(f"Bits:               {args.bits}")
    print(f"Group Size:         {args.group_size}")
    print(f"Calibration Samples: {args.calibration_samples}")
    print(f"Symmetric:          {symmetric}")
    print("=" * 60)
    print()

    # Import llm-compressor
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError as e:
        print(f"Error: Failed to import llm-compressor: {e}")
        print("Install with: pip install llmcompressor")
        sys.exit(1)

    print("Creating quantization recipe...")

    # Build recipe based on bits
    recipe = f"""
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: {args.bits}
                        type: "int"
                        symmetric: {str(symmetric).lower()}
                        group_size: {args.group_size}
                        strategy: "group"
                    targets: ["Linear"]
"""

    print("✓ Recipe created\n")
    print("Starting quantization...")
    print("This may take 10-30 minutes depending on model size...\n")

    # Run oneshot quantization
    oneshot(
        model=model_path,
        dataset="open_platypus",  # Calibration dataset
        output_dir=output_dir,
        num_calibration_samples=args.calibration_samples,
        recipe=recipe,
        max_seq_length=2048,
        pad_to_max_length=False,
    )

    print("\n✓ Quantization complete\n")

    print("=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"\nQuantized model saved to: {output_dir}")
    print(f"\nThis {quant_name} model can now be used with:")
    print("  • TensorRT-LLM (via build_engine.py --quant awq/int8)")
    print("  • vLLM")
    print("  • llama.cpp (convert to GGUF)")
    if args.bits == 8:
        print("\nFor llama.cpp conversion:")
        print(f"  python convert_hf_to_gguf.py {output_dir} --outtype q8_0")


if __name__ == "__main__":
    main()
