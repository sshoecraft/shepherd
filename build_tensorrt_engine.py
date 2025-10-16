#!/usr/bin/env python3.12
"""
Build TensorRT-LLM engine from HuggingFace TinyLlama model
"""
import os
import json
import shutil
from pathlib import Path

print("=" * 70)
print("TinyLlama → TensorRT-LLM Engine Build")
print("=" * 70)

# Paths
model_dir = Path.home() / "models/TinyLlama-1.1B-Chat"
checkpoint_dir = Path.home() / "models/tinyllama-checkpoint"
engine_dir = Path.home() / "models/tinyllama-engine"

print(f"\n📁 Source HF model: {model_dir}")
print(f"📁 TRT checkpoint: {checkpoint_dir}")
print(f"📁 TRT engine: {engine_dir}")

# Create dirs
checkpoint_dir.mkdir(parents=True, exist_ok=True)
engine_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Load HF config and create TRT-LLM config
print("\n🔧 Step 1: Creating TensorRT-LLM checkpoint...")
with open(model_dir / "config.json") as f:
    hf_config = json.load(f)

# Create TRT-LLM config
trt_config = {
    "architecture": "LLaMAForCausalLM",
    "dtype": "float16",
    "logits_dtype": "float32",
    "hidden_size": hf_config["hidden_size"],
    "intermediate_size": hf_config["intermediate_size"],
    "num_hidden_layers": hf_config["num_hidden_layers"],
    "num_attention_heads": hf_config["num_attention_heads"],
    "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
    "vocab_size": hf_config["vocab_size"],
    "max_position_embeddings": hf_config["max_position_embeddings"],
    "hidden_act": hf_config["hidden_act"],
    "norm_epsilon": hf_config["rms_norm_eps"],
    "position_embedding_type": "rope_gpt_neox",
    "mapping": {
        "world_size": 1,
        "tp_size": 1,
        "pp_size": 1
    }
}

# Save TRT-LLM config
config_path = checkpoint_dir / "config.json"
with open(config_path, 'w') as f:
    json.dump(trt_config, f, indent=2)
print(f"✅ Saved TRT-LLM config to {config_path}")

# Step 2: Convert weights using TRT-LLM's built-in converter
print("\n🔄 Step 2: Converting weights...")
import subprocess

# Use the HF model directly - TRT-LLM can convert on the fly
cmd = [
    "python3.12", "-c",
    f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / '.local/lib/python3.12/site-packages'))

from tensorrt_llm.models.llama.convert import load_weights_from_hf_safetensors
from tensorrt_llm.models import LLaMAConfig
import json

# Load config
with open('{checkpoint_dir}/config.json') as f:
    config_dict = json.load(f)

# Create config object
config = LLaMAConfig(**config_dict)

# Convert weights
print('Converting weights from HuggingFace format...')
weights = load_weights_from_hf_safetensors('{model_dir}', config)
print(f'Loaded {{len(weights)}} weight tensors')

# Save weights
import torch
torch.save(weights, '{checkpoint_dir}/rank0.safetensors')
print('Weights saved!')
"""
]

result = subprocess.run(cmd, shell=False)
if result.returncode != 0:
    print("❌ Weight conversion failed")
    exit(1)

print("✅ Weights converted")

# Step 3: Build engine
print("\n🏗️  Step 3: Building TensorRT engine (this may take several minutes)...")
cmd = [
    "python3.12", "-m", "tensorrt_llm.commands.build",
    "--checkpoint_dir", str(checkpoint_dir),
    "--output_dir", str(engine_dir),
    "--max_batch_size", "1",
    "--max_input_len", "512",
    "--max_seq_len", "1024",
]

print(f"Running: {' '.join(cmd)}\n")
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("✅ SUCCESS! TensorRT engine built")
    print("=" * 70)
    print(f"\n📁 Engine: {engine_dir}")

    # List engine files
    print("\nEngine files:")
    for f in engine_dir.glob("*"):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
else:
    print("\n❌ Engine build failed")
    exit(1)
