# Shepherd Build Requirements

This document lists all dependencies required to build Shepherd from source.

## Core Build Dependencies

### Required for All Builds

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    libsqlite3-dev \
    libcurl4-openssl-dev \
    pkg-config
```

### Optional: Line Editing Support

```bash
# For readline support (recommended)
sudo apt-get install -y libreadline-dev

# Alternative: libedit
sudo apt-get install -y libedit-dev
```

## Backend-Specific Dependencies

### llama.cpp Backend (Default, Recommended)

**CPU-only build:**
```bash
# No additional dependencies required
# llama.cpp will be built automatically as a submodule
```

**GPU-accelerated build (CUDA):**
```bash
# NVIDIA CUDA Toolkit 12.0 or later
# Download from: https://developer.nvidia.com/cuda-downloads
# Or install via package manager if available

sudo apt-get install -y nvidia-cuda-toolkit
```

### API Backends (OpenAI, Anthropic, Gemini, Grok, Ollama)

Already satisfied by `libcurl4-openssl-dev` (listed above).

### TensorRT-LLM Backend (Optional, Advanced)

**Warning:** TensorRT-LLM has complex dependencies and requires significant setup. Only needed for maximum GPU performance with stateful KV cache.

```bash
# 1. NVIDIA CUDA Toolkit 13.0 or later (12.4 is too old for TensorRT-LLM 1.1.0+)
#    DO NOT use Debian's nvidia-cuda-toolkit package (version 12.4)
#    Install from NVIDIA's repository:

cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0

# Add CUDA 13.0 to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0

# 2. TensorRT 10.13.3 for Debian 12 with CUDA 12.x/13.x
#    Requires NVIDIA Developer Account
#    Download from: https://developer.nvidia.com/tensorrt
#    Get: TensorRT 10.13.3 GA for Debian 12 and CUDA 12.0 to 12.9 (DEB local repo)
#    File: nv-tensorrt-local-repo-debian12-10.13.3-cuda-12.9_1.0-1_amd64.deb

sudo dpkg -i ~/nv-tensorrt-local-repo-debian12-10.13.3-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-debian12-10.13.3-cuda-12.9/nv-tensorrt-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y tensorrt libnvinfer10 libnvinfer-dev libnvinfer-plugin10 libnvinfer-plugin-dev
sudo apt-get install -y libnvonnxparsers10 libnvonnxparsers-dev

# 3. MPI and UCX (for multi-GPU support, optional for single GPU)
sudo apt-get install -y libopenmpi-dev openmpi-bin libucx-dev libucx0 ucx-utils

# 4. Python 3.12 (for build system only, NOT runtime)
#    Debian Trixie ships Python 3.13 which is too new
#    TensorRT-LLM requires Python 3.10, 3.11, or 3.12
#    Build Python 3.12 from source:

cd /tmp
wget https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tgz
tar -xf Python-3.12.8.tgz
cd Python-3.12.8
./configure --enable-optimizations --prefix=/usr/local --with-ensurepip=install
make -j8
sudo make altinstall  # Use altinstall to avoid overwriting system python3

# Verify installation
/usr/local/bin/python3.12 --version  # Should show Python 3.12.8

# 5. Fix TensorRT-LLM dependency conflicts
#    The requirements.txt has conflicting triton versions
cd ~/src/shepherd/TensorRT-LLM
sed -i 's/triton==3.3.1/triton==3.4.0/' requirements.txt
```

## Build Instructions

### Standard Build (llama.cpp + API backends)

```bash
# Clone repository
git clone https://github.com/yourusername/shepherd.git
cd shepherd
git submodule update --init --recursive

# Build llama.cpp first
cd llama.cpp
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)
cd ..

# Build Shepherd
cmake -B build
cmake --build build -j$(nproc)

# Install (optional)
sudo cmake --install build --prefix /usr/local
```

### Build with TensorRT-LLM (Advanced)

```bash
# After installing all TensorRT-LLM dependencies above:

# Build TensorRT-LLM C++ libraries using Python build script
# (generates required headers like fmha_cubin.h, then builds C++ libs)
cd TensorRT-LLM
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0

# Use Python 3.12 build script with --cpp_only flag
# This builds C++ libraries with NO Python runtime dependency
/usr/local/bin/python3.12 scripts/build_wheel.py --cpp_only --build_type=Release --no-venv

# The C++ libraries will be in: TensorRT-LLM/cpp/build/tensorrt_llm/
# Runtime has NO Python dependency - just pure C++ .so files

cd ..

# Build Shepherd with TensorRT-LLM enabled
cmake -B build -DENABLE_TENSORRT=ON
cmake --build build -j$(nproc)
```

## Minimum System Requirements

- **OS:** Linux (tested on Debian Trixie, Ubuntu 22.04+)
- **RAM:** 8GB minimum, 16GB+ recommended
- **Disk:** 10GB free space for build artifacts
- **Compiler:** GCC 11+ or Clang 14+
- **CMake:** 3.20 or later

## GPU Requirements (Optional)

### For llama.cpp with CUDA:
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- CUDA 11.0 or later

### For TensorRT-LLM:
- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer, e.g., RTX 3090 is sm_86)
- CUDA 13.0 or later (12.4 is too old)
- 8GB+ VRAM recommended
- Python 3.12 for build process only (NOT needed at runtime)

## Installed Library Locations

After `make install`, libraries are installed to:
```
/usr/local/bin/shepherd           # Main executable
/usr/local/lib/libllama.so        # llama.cpp libraries
/usr/local/lib/libggml*.so        # GGML libraries
```

The executable uses RPATH (`$ORIGIN/../lib`) to find libraries, so no `LD_LIBRARY_PATH` configuration is needed.

## Troubleshooting

### Missing header files during build
```bash
# Install development packages
sudo apt-get install -y build-essential linux-headers-$(uname -r)
```

### CMake version too old
```bash
# Add Kitware APT repository for latest CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt-get update
sudo apt-get install -y cmake
```

### CUDA not found
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# If not found, add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Docker Alternative (Not Recommended by User)

While Docker containerization simplifies dependency management, this project is designed to build natively. Docker is **NOT** required or recommended for Shepherd.

## Questions or Issues?

Please file an issue on GitHub with:
- Your OS and version (`uname -a`)
- CMake version (`cmake --version`)
- Compiler version (`gcc --version` or `clang --version`)
- Full build log
