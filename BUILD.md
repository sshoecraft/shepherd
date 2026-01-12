# Building Shepherd

## Prerequisites

**Required**:
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- pkg-config
- OpenSSL
- SQLite3
- libcurl
- ncurses
- nlohmann/json (bundled)
- Rust 1.65+ and Cargo (only required for TensorRT backend)

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential g++ cmake pkg-config \
    libssl-dev libcurl4-openssl-dev libsqlite3-dev libncurses-dev
```

**For running tests (optional):**
```bash
sudo apt-get install libgtest-dev
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

**Optional** (for local inference):
- llama.cpp (included as submodule, requires CUDA Toolkit for GPU support)
- TensorRT-LLM (for NVIDIA GPU optimization)
- CUDA Toolkit 11.8+ (for GPU support in llama.cpp backend)

**CUDA Toolkit** (for GPU support with llama.cpp):

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

## Build Instructions

```bash
# 1. Clone repository
git clone https://github.com/sshoecraft/shepherd.git
cd shepherd

# 2. Initialize submodules (llama.cpp and tokenizers)
git submodule update --init --recursive

# If tokenizers submodule wasn't initialized (older clones):
# git submodule add https://github.com/huggingface/tokenizers.git tokenizers

# Alternative: Clone with submodules in one step
# git clone --recursive https://github.com/sshoecraft/shepherd.git

# 3. Apply replxx build fixes (C++20 compatibility and library naming)
cd vendor/replxx
git apply ../../patches/replxx-build-fixes.patch
cd ../..

# 4. Build llama.cpp (if using llamacpp backend)
cd llama.cpp

# Apply Shepherd patches (KV cache eviction callbacks + layer assignment fix)
git apply ../patches/llama.patch

# Build llama.cpp
mkdir -p build && cd build

# With NVIDIA GPU support (requires CUDA Toolkit)
cmake .. -DGGML_CUDA=ON

# Without GPU (CPU only)
# cmake ..

make -j$(nproc)
cd ../..

# 5. Install TensorRT-LLM (ONLY needed for TensorRT backend)
# Skip steps 5-6 if you're only using llama.cpp backend

# Create Python virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install TensorRT-LLM from pip (includes all dependencies)
# This provides the TensorRT-LLM C++ libraries and headers
pip install tensorrt_llm

# Verify installation
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# 6. Build tokenizers library (needed for TensorRT backend)
cd tokenizers/tokenizers
cargo build --release --features=capi

# Copy libraries to lib/ directory
mkdir -p ../../lib
cp target/release/libtokenizers_c.a ../../lib/
cp target/release/libtokenizers_cpp.a ../../lib/
cd ../..

# 7. Build Shepherd
mkdir -p build && cd build

# For llama.cpp only (no TensorRT, no Rust/Python needed):
cmake .. -DENABLE_LLAMACPP=ON -DENABLE_TENSORRT=OFF

# For TensorRT (requires steps 5-6 completed):
# Make sure virtual environment is activated!
# source ~/venv/bin/activate
# cmake .. -DENABLE_TENSORRT=ON -DENABLE_LLAMACPP=ON

make -j$(nproc)

# 8. Run
./shepherd --model /path/to/model.gguf
```

## TensorRT Backend Requirements

The TensorRT backend requires several components to be built and installed:

### 1. **TensorRT-LLM Python Package**
   - Provides the C++ libraries (`libtensorrt_llm.so`, `libnvinfer_plugin_tensorrt_llm.so`)
   - Provides C++ headers (executor API, plugins, etc.)
   - Install via: `pip install tensorrt_llm`
   - CMake auto-detects installation from Python package

### 2. **tokenizers-cpp Library**
   - Rust-based tokenization library for accurate token counting
   - Built from `tokenizers/` submodule
   - Requires: Rust 1.65+ and Cargo
   - Produces: `libtokenizers_c.a` and `libtokenizers_cpp.a`
   - Must be placed in `lib/` directory

### 3. **NVIDIA Libraries** (installed automatically with TensorRT-LLM)
   - TensorRT 10.x
   - NCCL 2.x (for multi-GPU support)
   - CUDA Runtime

### 4. **MPI** (for multi-GPU models)
   - Required for models that use tensor parallelism
   - Install: `sudo apt install libopenmpi-dev openmpi-bin openmpi-common`
   - CMake automatically finds MPI when `ENABLE_TENSORRT=ON`

### Build Process Summary

```bash
# Install TensorRT-LLM
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install tensorrt_llm

# Build tokenizers library
cd tokenizers/tokenizers
cargo build --release --features=capi
mkdir -p ../../lib
cp target/release/libtokenizers_{c,cpp}.a ../../lib/
cd ../..

# Build Shepherd with TensorRT
mkdir -p build && cd build
cmake .. -DENABLE_TENSORRT=ON -DENABLE_LLAMACPP=ON
make -j$(nproc)
```

### How CMake Finds TensorRT Components

The CMakeLists.txt uses Python to locate TensorRT-LLM installation:

1. **Headers**: `python -c 'import tensorrt_llm; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), "include"))'`
2. **Libraries**: `python -c 'import tensorrt_llm; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs"))'`
3. **TensorRT**: `python -c 'import tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))'`
4. **NCCL**: `python -c 'from nvidia import nccl; print(os.path.join(os.path.dirname(nccl.__file__), "lib"))'`

If Python auto-detection fails, CMake falls back to hardcoded paths in `~/venv/lib/python3.*/site-packages/`.

### Verifying TensorRT Build

After building, verify all components are linked correctly:

```bash
# Check that TensorRT libraries are found
ldd build/shepherd | grep tensorrt
# Should show: libtensorrt_llm.so => /path/to/venv/.../libtensorrt_llm.so

# Check that tokenizers are statically linked
nm build/shepherd | grep tokenizers
# Should show tokenizers symbols

# Test TensorRT backend
./build/shepherd --backend tensorrt --model /path/to/tensorrt/model
```

## CMake Options

### Backend Selection

```bash
# Build with llama.cpp support only (no Rust/Python needed)
cmake -DENABLE_LLAMACPP=ON -DENABLE_TENSORRT=OFF ..

# Build with TensorRT-LLM support (requires TensorRT-LLM + tokenizers)
cmake -DENABLE_TENSORRT=ON -DENABLE_LLAMACPP=OFF ..

# Build with both backends (recommended)
cmake -DENABLE_LLAMACPP=ON -DENABLE_TENSORRT=ON ..

# Disable API backends (OpenAI, Anthropic, etc.)
cmake -DENABLE_API_BACKENDS=OFF ..
```

### Build Types

CMake defaults to **Debug** build (matching Makefile behavior):

```bash
# Debug build (default) - same as Makefile
# Flags: -g -O0 -D_DEBUG
cmake ..

# Explicit debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build with optimizations
# Flags: -O3 -march=native -DNDEBUG
cmake -DCMAKE_BUILD_TYPE=Release ..

# Release with debug symbols
# Flags: -g -O2 -DNDEBUG
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

**Build Type Comparison:**

| Build Type | Flags | Use Case |
|------------|-------|----------|
| Debug (default) | `-g -O0 -D_DEBUG` | Development, debugging with gdb |
| Release | `-O3 -march=native -DNDEBUG` | Production, maximum performance |
| RelWithDebInfo | `-g -O2 -DNDEBUG` | Production with profiling |

**Note**: The Makefile always uses Debug flags (`-g -O0 -D_DEBUG`).

## Building and Running Tests

Shepherd includes a comprehensive test suite using Google Test. Tests are optional and excluded from the default build.

### Prerequisites

Install Google Test:

```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev
```

### Enabling Tests

Add `TESTS=ON` to your `~/.shepherd_opts` file:

```bash
echo "TESTS=ON" >> ~/.shepherd_opts
```

Or pass it directly to cmake:

```bash
cmake -DBUILD_TESTS=ON ..
```

### Building Tests

```bash
# Configure with tests enabled
make gconfig   # Debug build with tests (if TESTS=ON in ~/.shepherd_opts)

# Build main project
make

# Build test executables
cd build && make test_unit test_tools
```

### Running Tests

```bash
# Run all unit tests
./build/tests/test_unit

# Run tool tests
./build/tests/test_tools

# Run specific test suite
./build/tests/test_unit --gtest_filter="ConfigTest.*"

# Run with verbose output
./build/tests/test_unit --gtest_output=xml:test_results.xml

# List available tests
./build/tests/test_unit --gtest_list_tests
```

### Test Categories

| Executable | Description |
|------------|-------------|
| `test_unit` | Core tests (config, session, scheduler, providers, message, SSE parser) |
| `test_tools` | Tool system tests (filesystem, command, memory, JSON) |

### Provider Tests

Provider tests are part of `test_unit` because providers are core Shepherd functionality. The tests automatically:
- Load all providers from `~/.config/shepherd/providers/`
- Test connectivity and basic generation for each configured provider
- Skip provider types that aren't configured
- Test provider switching if you have multiple providers

### Test Documentation

For detailed test cases and manual testing procedures, see:
- `docs/testing.md` - Complete test plan
- `docs/testing_manual_checklist.md` - Manual testing checklist
