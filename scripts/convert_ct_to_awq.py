#!/usr/bin/env python3
"""
Convert compressed-tensors format to AutoAWQ format

compressed-tensors (llmcompressor/vLLM) -> AutoAWQ (TRT-LLM compatible)

Both formats are INT4 with group_size 128, just packed differently.
No GPU needed - just repacking already-quantized weights.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def convert_packed_weights(weight_packed, weight_scale, weight_zero_point, weight_shape):
    """Convert compressed-tensors packed format to AutoAWQ format

    compressed-tensors: weight_packed is [out_features, in_features/8]
                       each int32 contains 8 consecutive int4 values along input dim
    AutoAWQ: qweight is [in_features, out_features/8]
             each int32 contains 8 consecutive int4 values along output dim
    """
    out_features = weight_shape[0].item()
    in_features = weight_shape[1].item()

    # Unpack int4 from int32 (8 values per int32)
    # weight_packed is [out, in/8], each int32 has 8 consecutive in_features
    unpacked = []
    for i in range(8):
        unpacked.append((weight_packed >> (i * 4)) & 0xF)

    # Stack and reshape to get [out_features, in_features]
    # Stack gives [out, in/8, 8], reshape gives [out, in]
    stacked = torch.stack(unpacked, dim=2)
    weight_int4 = stacked.reshape(out_features, in_features)

    # Transpose to [in_features, out_features]
    weight_int4_t = weight_int4.t().contiguous()

    # Repack for AutoAWQ: [in_features, out_features/8]
    # Pack with AutoAWQ interleaved order - value[i] goes to nibble AWQ_ORDER[i]
    AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    qweight = torch.zeros(in_features, out_features // 8, dtype=torch.int32)
    for i in range(8):
        qweight |= (weight_int4_t[:, i::8].to(torch.int32) << (AWQ_ORDER[i] * 4))

    # Scales: [out, groups] -> [groups, out] for QKV concatenation along dim 1
    scales = weight_scale.t().contiguous().to(torch.float16)

    # Zeros: [out/8, groups] packed int32 -> [groups, out/8] packed with AWQ order
    # First unpack CT zeros
    z_unpacked = []
    for i in range(8):
        z_unpacked.append((weight_zero_point >> (i * 4)) & 0xF)
    z_stacked = torch.stack(z_unpacked, dim=2)  # [out/8, groups, 8]
    zeros_full = z_stacked.permute(1, 0, 2).reshape(-1, out_features)  # [groups, out]

    # Repack with AWQ order: [groups, out/8]
    qzeros = torch.zeros(zeros_full.shape[0], out_features // 8, dtype=torch.int32)
    for i in range(8):
        qzeros |= (zeros_full[:, i::8].to(torch.int32) << (AWQ_ORDER[i] * 4))

    return qweight, scales, qzeros


class ShardedLoader:
    """Load tensors from sharded safetensors files"""

    def __init__(self, input_dir, index):
        self.input_dir = Path(input_dir)
        self.weight_map = index['weight_map']
        self.handles = {}

    def get_tensor(self, name):
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"Tensor {name} not found in index")

        if shard not in self.handles:
            self.handles[shard] = safe_open(self.input_dir / shard, framework='pt')

        return self.handles[shard].get_tensor(name)

    def close(self):
        for h in self.handles.values():
            del h
        self.handles.clear()


def convert_model(input_dir, output_dir):
    """Convert all safetensors files from compressed-tensors to AWQ format"""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = input_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load index
    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print("Error: No model.safetensors.index.json found")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    # Find all quantized layers (those with weight_packed)
    quant_layers = set()
    all_keys = set(index['weight_map'].keys())

    for key in all_keys:
        if key.endswith('.weight_packed'):
            base = key[:-len('.weight_packed')]
            quant_layers.add(base)

    print(f"Found {len(quant_layers)} quantized layers")
    print(f"Total tensors: {len(all_keys)}")

    # Create loader
    loader = ShardedLoader(input_dir, index)

    # Convert all tensors
    converted_tensors = {}

    # First convert quantized layers
    print("\nConverting quantized layers...")
    for base_name in tqdm(sorted(quant_layers)):
        weight_packed = loader.get_tensor(f"{base_name}.weight_packed")
        weight_scale = loader.get_tensor(f"{base_name}.weight_scale")
        weight_zero_point = loader.get_tensor(f"{base_name}.weight_zero_point")
        weight_shape = loader.get_tensor(f"{base_name}.weight_shape")

        qweight, scales, qzeros = convert_packed_weights(
            weight_packed, weight_scale, weight_zero_point, weight_shape
        )

        converted_tensors[f"{base_name}.qweight"] = qweight
        converted_tensors[f"{base_name}.scales"] = scales
        converted_tensors[f"{base_name}.qzeros"] = qzeros

    # Then copy non-quantized tensors
    print("\nCopying non-quantized tensors...")
    skip_suffixes = ('.weight_packed', '.weight_scale', '.weight_zero_point', '.weight_shape')

    for key in tqdm(sorted(all_keys)):
        if any(key.endswith(s) for s in skip_suffixes):
            continue

        tensor = loader.get_tensor(key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
        converted_tensors[key] = tensor

    loader.close()

    # Save in multiple shards (keep under 5GB per shard)
    print(f"\nSaving {len(converted_tensors)} converted tensors...")

    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
    current_shard = {}
    current_size = 0
    shard_num = 1
    weight_map = {}

    sorted_keys = sorted(converted_tensors.keys())

    for key in tqdm(sorted_keys):
        tensor = converted_tensors[key]
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            # Save current shard
            shard_name = f"model-{shard_num:05d}-of-XXXXX.safetensors"
            save_file(current_shard, output_dir / shard_name)
            shard_num += 1
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    # Save final shard
    if current_shard:
        shard_name = f"model-{shard_num:05d}-of-XXXXX.safetensors"
        save_file(current_shard, output_dir / shard_name)

    # Rename shards with correct count
    total_shards = shard_num
    for i in range(1, total_shards + 1):
        old_name = output_dir / f"model-{i:05d}-of-XXXXX.safetensors"
        new_name = output_dir / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old_name.rename(new_name)

        # Update weight map
        with safe_open(new_name, framework='pt') as f:
            for k in f.keys():
                weight_map[k] = new_name.name

    # Write index
    total_size = sum(os.path.getsize(output_dir / f) for f in os.listdir(output_dir) if f.endswith('.safetensors'))
    index_out = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    with open(output_dir / "model.safetensors.index.json", 'w') as f:
        json.dump(index_out, f, indent=2)

    # Update config
    new_quant_config = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "version": "gemm"
    }
    config['quantization_config'] = new_quant_config
    if config.get('torch_dtype') == 'bfloat16':
        config['torch_dtype'] = 'float16'

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                  'tokenizer.model', 'vocab.json', 'merges.txt', 'generation_config.json',
                  'chat_template.jinja']:
        src = input_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")

    print(f"\nConversion complete!")
    print(f"Output: {output_dir}")
    print(f"Shards: {total_shards}")


def main():
    parser = argparse.ArgumentParser(description="Convert compressed-tensors to AutoAWQ format")
    parser.add_argument("input_dir", help="Input model directory (compressed-tensors format)")
    parser.add_argument("--output", "-o", help="Output directory (default: input_dir-awq)")
    args = parser.parse_args()

    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = args.output or f"{input_dir}-awq"
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    print("=" * 60)
    print("compressed-tensors -> AutoAWQ Converter")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    convert_model(input_dir, output_dir)


if __name__ == "__main__":
    main()
