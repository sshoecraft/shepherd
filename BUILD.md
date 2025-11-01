# Building Shepherd

## Prerequisites

**Required**:
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- SQLite3
- libcurl
- nlohmann/json
- Rust 1.65+ and Cargo (only required for TensorRT backend)

**Optional** (for local inference):
- llama.cpp (included as submodule, requires CUDA Toolkit for GPU support)
- TensorRT-LLM (for NVIDIA GPU optimization)
- CUDA Toolkit 11.8+ (for GPU support in llama.cpp backend)

## Build Instructions

```bash
# 1. Clone repository
git clone https://github.com/sshoecraft/shepherd.git
cd shepherd

# 2. Initialize submodules (llama.cpp and tokenizers)
git submodule update --init --recursive

# Alternative: Clone with submodules in one step
# git clone --recursive https://github.com/sshoecraft/shepherd.git

# 3. Build llama.cpp (if using llamacpp backend)
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

# 4. Build tokenizers library (ONLY needed for TensorRT backend)
# Skip this step if you're only using llama.cpp backend
cd tokenizers/tokenizers
cargo build --release --features=capi
cp target/release/libtokenizers_c.a ../../lib/
cp target/release/libtokenizers_cpp.a ../../lib/
cd ../..

# 5. Build Shepherd
mkdir -p build && cd build

# For llama.cpp only (no TensorRT, no Rust needed):
cmake .. -DENABLE_LLAMACPP=ON -DENABLE_TENSORRT=OFF

# For TensorRT (requires tokenizers library built in step 4):
# cmake .. -DENABLE_TENSORRT=ON -DENABLE_LLAMACPP=ON

make -j$(nproc)

# 6. Run
./shepherd --model /path/to/model.gguf
```

## CMake Options

```bash
# Build with llama.cpp support
cmake -DUSE_LLAMACPP=ON ..

# Build with TensorRT-LLM support
cmake -DUSE_TENSORRT=ON ..

# Build with all backends (default)
cmake -DUSE_LLAMACPP=ON -DUSE_TENSORRT=ON ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
```
