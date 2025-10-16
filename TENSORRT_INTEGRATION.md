# TensorRT-LLM Backend Integration

## Overview

The TensorRT backend has been implemented for Shepherd to enable GPU-accelerated inference using NVIDIA's TensorRT-LLM library. This backend follows the same architecture as the llama.cpp backend and provides similar functionality while leveraging TensorRT's optimizations.

## Architecture

The TensorRT backend consists of three main components:

### 1. TensorRTTokenizer (`backends/tensorrt.h`, `backends/tensorrt.cpp`)
- Provides tokenization interface (currently uses approximation)
- Note: TensorRT-LLM doesn't expose its tokenizer directly, so text-to-token conversion needs to be handled externally or with placeholder tokens
- **TODO**: Integrate with HuggingFace tokenizers or model-specific tokenizers

### 2. TensorRTContextManager (`backends/tensorrt.h`, `backends/tensorrt.cpp`)
- Manages conversation context
- Formats messages for inference
- Handles token counting and context window management
- Currently uses simple string concatenation; can be enhanced with proper chat templates

### 3. TensorRTBackend (`backends/tensorrt.h`, `backends/tensorrt.cpp`)
- Main backend interface implementation
- Uses TensorRT-LLM Executor API for inference
- Handles request queueing and response streaming
- Manages executor lifecycle (initialization, generation, shutdown)

## Key Features

### Implemented
- ✅ Executor-based inference using TensorRT-LLM's high-level API
- ✅ Streaming response support
- ✅ Context management and message tracking
- ✅ Tool system integration
- ✅ KV cache block reuse configuration
- ✅ Proper backend registration and factory pattern

### TODO / Known Limitations
- ⚠️ **Tokenization**: Currently uses placeholder tokens. Needs integration with actual tokenizer
- ⚠️ **Detokenization**: Token IDs are returned instead of text. Needs proper detokenization
- ⚠️ **Chat Templates**: Simple string concatenation. Should use model-specific chat templates
- ⚠️ **KV Cache Eviction**: Executor manages KV cache internally; eviction callback not yet implemented
- ⚠️ **Multi-GPU**: Not yet tested with multi-GPU configurations

## TensorRT-LLM API Usage

### Initialization
```cpp
tensorrt_llm::executor::ExecutorConfig config;
config.setMaxBeamWidth(1);

tensorrt_llm::executor::KvCacheConfig kvCacheConfig;
kvCacheConfig.setEnableBlockReuse(true);
config.setKvCacheConfig(kvCacheConfig);

auto* executor = new tensorrt_llm::executor::Executor(
    model_path,
    tensorrt_llm::executor::ModelType::kDECODER_ONLY,
    config
);
```

### Request and Response
```cpp
// Create request
tensorrt_llm::executor::Request request(
    input_tokens,        // std::vector<int32_t>
    max_tokens,          // int
    streaming,           // bool
    samplingConfig,      // SamplingConfig
    outputConfig         // OutputConfig
);

// Enqueue
uint64_t request_id = executor->enqueueRequest(request);

// Await responses
auto responses = executor->awaitResponses(request_id, timeout);

for (const auto& response : responses) {
    const auto& result = response.getResult();
    // result.outputTokenIds contains generated tokens
    // result.isFinal indicates completion
}
```

## CMake Configuration

### Enable TensorRT Backend
```bash
cmake -DENABLE_TENSORRT=ON ..
```

The CMakeLists.txt automatically:
1. Locates TensorRT-LLM installation
2. Finds tensorrt_llm and executor_static libraries
3. Links CUDA runtime and TensorRT
4. Sets up include paths
5. Configures RPATH for library resolution

### Required Libraries
- `tensorrt_llm`: Main TensorRT-LLM library
- `executor_static`: Executor API implementation
- CUDA runtime
- TensorRT (nvinfer)

## Building

### Prerequisites
1. Build TensorRT-LLM first:
```bash
cd TensorRT-LLM
# Follow TensorRT-LLM build instructions
# Apply patches if needed (see patches/tensorrt-llm-disable-userbuffers.patch)
```

2. Build Shepherd with TensorRT:
```bash
mkdir build
cd build
cmake -DENABLE_TENSORRT=ON ..
make -j$(nproc)
```

## Usage

```bash
./shepherd --backend tensorrt --model /path/to/tensorrt/engine
```

The model path should point to a TensorRT-LLM engine directory (output of `trtllm-build`).

## Integration with Main Loop

The TensorRT backend integrates seamlessly with Shepherd's main conversation loop:

1. **Context Building**: Messages are added via `add_user_message()`, `add_assistant_message()`, etc.
2. **Generation**: Main calls `generate()` which:
   - Gets formatted context
   - Tokenizes (currently placeholder)
   - Enqueues request to executor
   - Streams responses back
3. **Tool Execution**: Tool results are added back to context
4. **Memory Management**: Context manager tracks token usage

## Next Steps for Production

1. **Tokenization Integration**:
   - Link HuggingFace tokenizers library
   - Load tokenizer from model config
   - Implement proper encode/decode methods

2. **Model-Specific Templates**:
   - Extract chat template from model metadata
   - Apply proper formatting for different model types (Llama, Mistral, etc.)

3. **Performance Optimization**:
   - Tune sampling parameters
   - Configure optimal batch sizes
   - Enable tensor parallelism for multi-GPU

4. **Testing**:
   - Unit tests for tokenization
   - Integration tests with sample models
   - Benchmark against llama.cpp backend

5. **Documentation**:
   - Model preparation guide
   - Performance tuning guide
   - Troubleshooting common issues

## References

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [Executor API Reference](TensorRT-LLM/cpp/include/tensorrt_llm/executor/executor.h)
- [llama.cpp Backend Implementation](backends/llamacpp.cpp) - Reference implementation
