# Shepherd

Advanced RAG-enabled LLM System with hierarchical memory management.

## Overview

Shepherd is a C++ implementation of an LLM inference system with integrated Retrieval-Augmented Generation (RAG) capabilities. It features a sophisticated hierarchical memory management system that efficiently utilizes GPU VRAM, system RAM, and storage for optimal context window handling.

## Features

- **Dual Backend Support**: Works with both llama.cpp (development) and TensorRT-LLM (production)
- **Hierarchical Memory Management**: Three-tier memory system (GPU VRAM → System RAM → Storage)
- **RAG Integration**: Document retrieval and context enhancement
- **Optimized for NVIDIA GPUs**: Designed for dual RTX 3090 systems with 128GB RAM
- **Large Context Windows**: Support for 32K-256K+ token contexts depending on model size

## Memory Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  GPU VRAM   │    │ System RAM   │    │   Storage   │
│   (Hot)     │ ── │   (Warm)     │ ── │   (Cold)    │
│ Recent      │    │ Extended     │    │ Vector DB   │
│ Context     │    │ Context      │    │ Documents   │
└─────────────┘    └──────────────┘    └─────────────┘
```

## Build Requirements

### Development (macOS)
```bash
brew install llama.cpp cmake
```

### Production (Linux with NVIDIA GPUs)
```bash
# Install TensorRT-LLM (see NVIDIA documentation)
# Or fallback to llama.cpp with CUDA support
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
# Basic usage
./shepherd path/to/model.gguf

# Interactive mode
./shepherd path/to/model.gguf
> What is machine learning?
> quit
```

## Configuration

The system automatically detects available backends:
1. **TensorRT-LLM** (if available) - Maximum NVIDIA GPU performance
2. **llama.cpp** (fallback) - Portable CPU/GPU inference

Memory limits are automatically configured based on detected hardware.

## Architecture

- **main.cpp**: Application entry point and interactive loop
- **inference_engine**: Abstraction layer for llama.cpp/TensorRT-LLM
- **rag_system**: Document retrieval and prompt enhancement
- **memory_manager**: Hierarchical context and memory management

## Hardware Optimization

Designed for systems with:
- Dual NVIDIA RTX 3090 GPUs (48GB VRAM total)
- 128GB system RAM
- NVMe SSD storage

Supports context windows of 64K-256K+ tokens depending on model size.

## Development

The codebase uses system-installed inference libraries rather than vendoring them, allowing for easy switching between development (llama.cpp) and production (TensorRT-LLM) environments.

## License

[Specify license]