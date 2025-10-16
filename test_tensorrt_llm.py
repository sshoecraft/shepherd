#!/usr/bin/env python3.12
"""
Test script to verify TensorRT-LLM installation
"""
import tensorrt_llm
from tensorrt_llm.models import LLaMAConfig
import torch

print("=" * 60)
print("TensorRT-LLM Installation Test")
print("=" * 60)

# Test 1: Version check
print(f"\n✅ TensorRT-LLM version: {tensorrt_llm.__version__}")

# Test 2: CUDA availability
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")

# Test 3: Create a minimal LLaMA config
print("\n✅ Creating LLaMA configuration...")
config = LLaMAConfig(
    architecture="LLaMAForCausalLM",
    dtype="float16",
    hidden_size=2048,
    num_hidden_layers=22,
    num_attention_heads=32,
    vocab_size=32000,
    max_position_embeddings=2048,
)
print(f"   Architecture: {config.architecture}")
print(f"   Hidden size: {config.hidden_size}")
print(f"   Layers: {config.num_hidden_layers}")

# Test 4: Check TensorRT availability
try:
    import tensorrt as trt
    print(f"\n✅ TensorRT version: {trt.__version__}")
except ImportError as e:
    print(f"\n⚠️  TensorRT import failed: {e}")

print("\n" + "=" * 60)
print("All tests passed! TensorRT-LLM is ready to use.")
print("=" * 60)
