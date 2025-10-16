#!/usr/bin/env python3.12
"""
Build and run TinyLlama with TensorRT-LLM using the high-level API
"""
from pathlib import Path
import subprocess
import sys

print("=" * 70)
print("TinyLlama with TensorRT-LLM - Build & Run")
print("=" * 70)

model_dir = Path.home() / "models/TinyLlama-1.1B-Chat"
engine_dir = Path.home() / "models/tinyllama-engine"

engine_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Model: {model_dir}")
print(f"📁 Engine output: {engine_dir}")

# Build engine using trtllm-build with HF model
print("\n🏗️  Building TensorRT engine...")
print("This will take a few minutes on first run...\n")

cmd = [
    "trtllm-build",
    "--model_config", str(model_dir / "config.json"),
    "--max_batch_size", "1",
    "--max_input_len", "512",
    "--max_seq_len", "1024",
    "--output_dir", str(engine_dir),
]

# Try using trtllm-build if available
result = subprocess.run(["which", "trtllm-build"], capture_output=True)
if result.returncode != 0:
    # Fall back to python module
    print("Using: python3.12 -m tensorrt_llm.commands.build\n")
    cmd = [
        "python3.12", "-m", "tensorrt_llm.commands.build",
        "--model_config", str(model_dir / "config.json"),
        "--max_batch_size", "1",
        "--max_input_len", "512",
        "--max_seq_len", "1024",
        "--output_dir", str(engine_dir),
    ]

print(f"Command: {' '.join(cmd)}\n")
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✅ Engine built successfully!")

    # List files
    print(f"\nEngine files in {engine_dir}:")
    for f in sorted(engine_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size / (1024*1024)
            print(f"  {f.name:40s} {size:8.1f} MB")

    # Now run inference
    print("\n" + "=" * 70)
    print("🚀 Running inference test...")
    print("=" * 70)

    inference_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / '.local/lib/python3.12/site-packages'))

from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

print("\\n📥 Loading engine and tokenizer...")
engine_dir = Path.home() / "models/tinyllama-engine"
model_dir = Path.home() / "models/TinyLlama-1.1B-Chat"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

# Load TRT-LLM engine
runner = ModelRunner.from_dir(str(engine_dir), rank=0)

print("✅ Engine loaded!\\n")

# Run inference
prompt = "Once upon a time"
print(f"Prompt: '{prompt}'")
print("\\nGenerating...\\n")

input_ids = tokenizer.encode(prompt, return_tensors='pt')

outputs = runner.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:")
print("-" * 70)
print(generated_text)
print("-" * 70)
"""

    result = subprocess.run([
        "python3.12", "-c", inference_script
    ])

    if result.returncode == 0:
        print("\n✅ Inference successful!")
    else:
        print("\n⚠️  Inference test had issues, but engine was built")

else:
    print("\n❌ Engine build failed")
    sys.exit(1)
