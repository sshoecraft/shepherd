#!/bin/bash
# Build Shepherd with TensorRT backend only

set -e

echo "Building Shepherd with TensorRT-LLM backend..."

# Set CUDA compiler
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Use venv Python that has tensorrt_llm installed
export PYTHON_EXECUTABLE=~/venv/bin/python3

# TensorRT-LLM Python site-packages (has C++ headers)
TRT_INCLUDE=~/venv/lib/python3.12/site-packages

# Clean build
rm -rf build
mkdir -p build
cd build

# Configure with TensorRT backend enabled
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TENSORRT=ON \
  -DENABLE_API_BACKENDS=OFF \
  -DTensorRT_INCLUDE_DIR=$TRT_INCLUDE \
  -DPython3_EXECUTABLE=$PYTHON_EXECUTABLE

# Build
cmake --build . -j16

echo "Done! Binary: build/shepherd"
