# TensorRT-LLM Backend Setup

This document describes how to set up the TensorRT-LLM backend for Shepherd.

## Prerequisites

- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- NVIDIA driver 570+ (580+ recommended for TRT-LLM 1.2.x)
- CUDA Toolkit (version must match TRT-LLM requirements)
- Python 3.10 or 3.12
- OpenMPI (for multi-GPU engine builds with tensor parallelism)

### OpenMPI Installation

For building engines with tensor parallelism (tp > 1), OpenMPI is required:

```bash
sudo apt install openmpi-bin openmpi-common
```

## Step 1: Install TensorRT-LLM

Install from NVIDIA's PyPI:

```bash
pip install --extra-index-url https://pypi.nvidia.com tensorrt-llm==1.2.0rc4
```

Verify installation:

```bash
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

## Step 2: Versioned Headers

Shepherd includes bundled C++ headers for supported TensorRT-LLM versions in:

```
include/tensorrt-llm/<version>/tensorrt_llm/
```

Currently bundled versions:
- `1.0.0`
- `1.2.0rc3`

CMake automatically detects your installed TRT-LLM version and selects the matching headers.

### Adding Headers for a New Version

If your installed TRT-LLM version doesn't have bundled headers:

```bash
# Clone or update TensorRT-LLM source
cd ~/src/TensorRT-LLM
git fetch --tags
git checkout v<version>  # e.g., v1.2.0rc4

# Create versioned header directory
cd /path/to/shepherd
mkdir -p include/tensorrt-llm/<version>/tensorrt_llm

# Copy headers
cp -r ~/src/TensorRT-LLM/cpp/include/tensorrt_llm/* include/tensorrt-llm/<version>/tensorrt_llm/
```

## Step 3: Install CUDA Toolkit

TRT-LLM 1.2.x requires CUDA 13. Check available cuBLAS:

```bash
ldconfig -p | grep cublasLt
```

If you only have cuBLAS 12, install CUDA 13:

```bash
# Debian/Ubuntu (using Debian 12 packages on Debian 13)
wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-debian12-13-0-local_13.0.2-580.95.05-1_amd64.deb
sudo dpkg -i cuda-repo-debian12-13-0-local_13.0.2-580.95.05-1_amd64.deb
sudo cp /var/cuda-repo-debian12-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install cuda-toolkit-13-0

# Add to ldconfig
echo "/usr/local/cuda-13.0/targets/x86_64-linux/lib" | sudo tee /etc/ld.so.conf.d/cuda-13.conf
sudo ldconfig
```

## Step 4: Build Shepherd with TensorRT

Configure and build:

```bash
cd /path/to/shepherd
mkdir -p build && cd build
cmake .. -DENABLE_TENSORRT=ON
make -j$(nproc)
```

## Step 5: Build TensorRT Engine

Convert your model to TensorRT format using `trtllm-build`. Example for an AWQ model:

```bash
trtllm-build \
    --checkpoint_dir /path/to/converted/checkpoint \
    --output_dir /path/to/engine \
    --gemm_plugin auto \
    --max_batch_size 1 \
    --max_input_len 32768 \
    --max_seq_len 32768
```

## Troubleshooting

### Linker errors about SamplingConfig or other TRT-LLM classes

Headers don't match the installed library version. Either:
- Your TRT-LLM version doesn't have bundled headers - add them (see Step 2)
- CMake cached wrong version - delete build/ and reconfigure

### CMake says "No headers found for TensorRT-LLM X.Y.Z"

You need to add headers for that version. See "Adding Headers for a New Version" in Step 2.

### libcublasLt.so.13: cannot open shared object file

Install CUDA 13 toolkit (Step 3).

### Multi-GPU P2P Issues

For multi-GPU setups with peer-to-peer memory access, you may need custom NVIDIA kernel modules with P2P support. See the tinygrad P2P fork: https://github.com/tinygrad/open-gpu-kernel-modules

## Version Compatibility Matrix

| TRT-LLM Version | CUDA Version | cuBLAS | Headers Bundled | Notes |
|-----------------|--------------|--------|-----------------|-------|
| 1.0.0           | 12.x         | 12     | Yes             | Stable, CUDA 12 |
| 1.1.0rc5        | 12.x         | 12     | No              | |
| 1.2.0rc3        | 13.x         | 13     | Yes             | Requires CUDA 13 |
| 1.2.0rc4        | 13.x         | 13     | No              | Requires CUDA 13 |

## Useful Commands

Check TRT-LLM version:
```bash
pip show tensorrt-llm | grep Version
```

Check CUDA/cuBLAS:
```bash
nvcc --version
ldconfig -p | grep cublas
```

Check GPU driver:
```bash
nvidia-smi
```
