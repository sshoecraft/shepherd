#!/usr/bin/env python3.12
"""
Convert HuggingFace TinyLlama to TensorRT-LLM and build engine
"""
import os
import sys
from pathlib import Path

# Add TensorRT-LLM to path
sys.path.insert(0, str(Path.home() / '.local/lib/python3.12/site-packages'))

from tensorrt_llm.models.llama.convert import load_weights_from_hf_safetensors
from tensorrt_llm.models import LLaMAConfig

print("=" * 70)
print("Llama 3.1 8B → TensorRT-LLM Conversion & Engine Build")
print("=" * 70)

# Paths
model_dir = Path.home() / "models/Llama-3.1-8B-Instruct-FP8"
output_dir = Path.home() / "models/llama-3.1-8b-trt-checkpoint"
engine_dir = Path.home() / "models/llama-3.1-8b-engine"

print(f"\n📁 Source model: {model_dir}")
print(f"📁 Output checkpoint: {output_dir}")
print(f"📁 Engine output: {engine_dir}")

# Create output directories
output_dir.mkdir(parents=True, exist_ok=True)
engine_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Create TensorRT-LLM config from HF config
print("\n🔧 Step 1: Creating TensorRT-LLM configuration...")
import json
with open(model_dir / "config.json") as f:
    hf_config = json.load(f)

print(f"   Model architecture: {hf_config.get('architectures', ['unknown'])[0]}")
print(f"   Hidden size: {hf_config['hidden_size']}")
print(f"   Layers: {hf_config['num_hidden_layers']}")
print(f"   Attention heads: {hf_config['num_attention_heads']}")
print(f"   Vocab size: {hf_config['vocab_size']}")

config = LLaMAConfig(
    architecture="LLaMAForCausalLM",
    dtype="float16",
    logits_dtype="float32",
    hidden_size=hf_config['hidden_size'],
    intermediate_size=hf_config['intermediate_size'],
    num_hidden_layers=hf_config['num_hidden_layers'],
    num_attention_heads=hf_config['num_attention_heads'],
    num_key_value_heads=hf_config.get('num_key_value_heads', hf_config['num_attention_heads']),
    vocab_size=hf_config['vocab_size'],
    max_position_embeddings=hf_config['max_position_embeddings'],
    hidden_act=hf_config['hidden_act'],
    norm_epsilon=hf_config['rms_norm_eps'],
)

print("✅ Configuration created")

# Step 2: Load and convert weights
print("\n🔄 Step 2: Converting weights from HuggingFace format...")
try:
    weights = load_weights_from_hf_safetensors(str(model_dir), config)
    print(f"✅ Loaded {len(weights)} weight tensors")
except Exception as e:
    print(f"❌ Weight conversion failed: {e}")
    print("\nTrying alternative: direct checkpoint build...")
    # We'll use trtllm-build directly on HF model
    import subprocess

    cmd = [
        "python3.12", "-m", "tensorrt_llm.commands.build",
        "--model_config", str(model_dir / "config.json"),
        "--checkpoint_dir", str(model_dir),
        "--output_dir", str(engine_dir),
        "--max_batch_size", "1",
        "--max_input_len", "512",
        "--max_seq_len", "1024",
        "--max_beam_width", "1",
    ]

    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✅ Engine built successfully!")
        print(f"📁 Engine location: {engine_dir}")
        sys.exit(0)
    else:
        print("\n❌ Engine build failed")
        sys.exit(1)

# Save checkpoint
print(f"\n💾 Step 3: Saving checkpoint to {output_dir}...")
import torch
torch.save({
    'config': config.to_dict(),
    'weights': weights
}, output_dir / "model.pth")

# Also save config.json separately for trtllm-build
with open(output_dir / "config.json", "w") as f:
    json.dump(config.to_dict(), f, indent=2)
print("✅ Checkpoint saved")

# Step 4: Build TensorRT engine
print("\n🏗️  Step 4: Building TensorRT engine...")
import subprocess

cmd = [
    "python3.12", "-m", "tensorrt_llm.commands.build",
    "--checkpoint_dir", str(output_dir),
    "--output_dir", str(engine_dir),
    "--max_batch_size", "1",
    "--max_input_len", "512",
    "--max_seq_len", "1024",
    "--max_beam_width", "1",
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("✅ SUCCESS! TensorRT engine built")
    print("=" * 70)
    print(f"\n📁 Engine location: {engine_dir}")
    print("\nYou can now use this engine for inference!")
else:
    print("\n❌ Engine build failed")
    sys.exit(1)
