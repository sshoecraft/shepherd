# Building Shepherd

## Prerequisites

**Required** (all build configurations):
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- pkg-config
- OpenSSL
- SQLite3
- libcurl
- ncurses
- nlohmann/json (bundled)

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential g++ cmake pkg-config \
    libssl-dev libcurl4-openssl-dev libsqlite3-dev libncurses-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc-c++ cmake pkgconfig \
    openssl-devel libcurl-devel sqlite-devel ncurses-devel
```

**macOS (Homebrew):**
```bash
brew install cmake pkg-config openssl curl sqlite ncurses
```

---

## Quick Build (API-Only / Cloud Providers)

If you are only using cloud API backends (OpenAI, Anthropic, Gemini, Azure, Grok, Ollama), this is all you need:

```bash
git clone --recursive https://github.com/sshoecraft/shepherd.git
cd shepherd
make
```

That's it. The Makefile handles everything: submodule initialization, replxx patching, cmake configuration, and building. It will check for required dependencies and tell you what's missing.

If you already cloned without `--recursive`:
```bash
git submodule update --init
make
```

---

## Build with llama.cpp (Local GPU Inference)

Requires NVIDIA GPU with CUDA Toolkit installed.

### Install CUDA Toolkit

Ubuntu/Debian:
```bash
# Add NVIDIA CUDA repository (Ubuntu 24.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA toolkit (nvcc + development libraries)
sudo apt install cuda-nvcc-12-8 cuda-libraries-dev-12-8

# Add to PATH (or add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh
```

For other Ubuntu versions, replace `ubuntu2404` with your version (e.g., `ubuntu2204` for 22.04).
See https://developer.nvidia.com/cuda-downloads for other distributions.

### Build llama.cpp and Shepherd

```bash
git clone --recursive https://github.com/sshoecraft/shepherd.git
cd shepherd

# Apply Shepherd patches to llama.cpp (KV cache eviction callbacks + layer assignment fix)
cd llama.cpp
git apply ../patches/llama.patch

# Build llama.cpp with CUDA
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON
make -j$(nproc)
cd ../..

# Build Shepherd with llama.cpp enabled
echo "LLAMACPP=ON" >> ~/.shepherd_opts
make
```

To build llama.cpp CPU-only (no GPU), use `cmake ..` instead of `cmake .. -DGGML_CUDA=ON`.

---

## Build with TensorRT-LLM (NVIDIA Optimized Inference)

Requires NVIDIA GPU, CUDA Toolkit, Rust 1.65+, and Python 3.

### Additional prerequisites

- Rust 1.65+ and Cargo
- Python 3 with pip

### Build steps

```bash
git clone --recursive https://github.com/sshoecraft/shepherd.git
cd shepherd

# 1. Install TensorRT-LLM
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install tensorrt_llm

# Verify installation
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# 2. Build tokenizers library
cd tokenizers/tokenizers
cargo build --release --features=capi
mkdir -p ../../lib
cp target/release/libtokenizers_c.a ../../lib/
cp target/release/libtokenizers_cpp.a ../../lib/
cd ../..

# 3. Build Shepherd with TensorRT enabled
echo "TENSORRT=ON" >> ~/.shepherd_opts
make
```

### How CMake Finds TensorRT Components

The CMakeLists.txt uses Python to locate TensorRT-LLM installation:

1. **Headers**: `python -c 'import tensorrt_llm; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), "include"))'`
2. **Libraries**: `python -c 'import tensorrt_llm; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs"))'`
3. **TensorRT**: `python -c 'import tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))'`
4. **NCCL**: `python -c 'from nvidia import nccl; print(os.path.join(os.path.dirname(nccl.__file__), "lib"))'`

If Python auto-detection fails, CMake falls back to hardcoded paths in `~/venv/lib/python3.*/site-packages/`.

### Verifying TensorRT Build

```bash
# Check that TensorRT libraries are found
ldd build/shepherd | grep tensorrt

# Check that tokenizers are statically linked
nm build/shepherd | grep tokenizers

# Test TensorRT backend
./build/shepherd --backend tensorrt --model /path/to/tensorrt/model
```

### MPI (for multi-GPU models)

Required for models that use tensor parallelism:
```bash
sudo apt install libopenmpi-dev openmpi-bin openmpi-common
```

CMake automatically finds MPI when `ENABLE_TENSORRT=ON`.

---

## Makefile Reference

The Makefile is the primary build interface. It wraps cmake and handles configuration automatically.

| Command | Description |
|---------|-------------|
| `make` | Build (auto-configures on first run) |
| `make config` | Reconfigure cmake (reads `~/.shepherd_opts`) |
| `make clean` | Clean build artifacts |
| `make distclean` | Remove entire build directory |
| `make install` | Install to system |

### Build Options (`~/.shepherd_opts`)

Create or edit `~/.shepherd_opts` to set build options:

```bash
LLAMACPP=ON       # Enable llama.cpp backend (requires llama.cpp built separately)
TENSORRT=ON       # Enable TensorRT backend (requires TensorRT-LLM + tokenizers)
POSTGRES=ON       # Enable PostgreSQL RAG backend
TESTS=ON          # Build test executables
RELEASE=yes       # Release build (optimized, no debug symbols)
```

All default to `OFF` except `RELEASE` which defaults to `NO` (debug build).

### CMake Options (Direct)

```bash
# Backend selection
cmake -DENABLE_LLAMACPP=ON -DENABLE_TENSORRT=OFF ..
cmake -DENABLE_TENSORRT=ON -DENABLE_LLAMACPP=ON ..
cmake -DENABLE_API_BACKENDS=OFF ..

# Build types
cmake -DCMAKE_BUILD_TYPE=Debug ..           # -g -O0 -D_DEBUG (default)
cmake -DCMAKE_BUILD_TYPE=Release ..         # -O3 -march=native -DNDEBUG
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..  # -g -O2 -DNDEBUG
```

---

## Building and Running Tests

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev
```

### Build and Run

```bash
echo "TESTS=ON" >> ~/.shepherd_opts
make config
make

# Run tests
./build/tests/test_unit
./build/tests/test_tools

# Run specific test suite
./build/tests/test_unit --gtest_filter="ConfigTest.*"

# List available tests
./build/tests/test_unit --gtest_list_tests
```

### Test Executables

| Executable | Description |
|------------|-------------|
| `test_unit` | Core tests (config, session, scheduler, providers, message, SSE parser) |
| `test_tools` | Tool system tests (filesystem, command, memory, JSON) |

See [docs/testing.md](docs/testing.md) for the full test plan.
