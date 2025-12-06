# llama.cpp Backend Architecture

## Overview

The llama.cpp backend (`llamacpp.cpp`) provides inference using the llama.cpp library. It supports GGUF model files with CPU and GPU acceleration (CUDA, Metal), multi-GPU configurations, and maintains a stateful KV cache for efficient multi-turn conversations.

## Model Loading

### Model Path Resolution

The backend constructs the full model path from config:

```cpp
// If absolute path, use directly
if (model_filename[0] == '/') {
    full_model_path = model_filename;
} else {
    // Combine model_path directory with filename
    full_model_path = model_path / model_filename;
}
```

### GPU Layer Configuration

Priority for GPU layers:
1. `GGML_N_GPU_LAYERS` environment variable
2. `gpu_layers` config setting
3. Auto mode (`-1` or unset): loads all layers to GPU

```cpp
if (gpu_layers >= 0) {
    model_params.n_gpu_layers = gpu_layers;
} else {
    model_params.n_gpu_layers = INT32_MAX;  // Auto: llama.cpp caps at actual count
}
```

### Multi-GPU Support

Two parallelism modes available:

1. **Pipeline Parallelism (PP)** - `pipeline_parallel` config
   - Uses `LLAMA_SPLIT_MODE_LAYER`
   - Distributes layers across GPUs
   - No peer-to-peer GPU communication needed

2. **Tensor Parallelism (TP)** - `tensor_parallel` config
   - Uses `LLAMA_SPLIT_MODE_ROW`
   - Splits individual tensors across GPUs
   - Requires GPU peer-to-peer support

```cpp
if (tensor_parallel > 1) {
    model_params.split_mode = LLAMA_SPLIT_MODE_ROW;
} else {
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
}
```

## Stateful KV Cache

### Design Philosophy

Unlike typical inference where each request rebuilds context, shepherd maintains a persistent KV cache across turns:

- Messages are tokenized and added to KV cache incrementally
- The cache persists between user messages
- Only new tokens need to be processed for each turn
- Dramatically reduces latency for multi-turn conversations

### Token Tracking

Each message in the session tracks its token count:

```cpp
struct Message {
    int tokens;  // Actual tokens in KV cache for this message
};
```

The backend verifies KV cache state matches session state:

```cpp
int expected = sum of message.tokens;
int actual = llama_memory_seq_pos_max(mem, 0) + 1;
// These should match
```

### Eviction Strategy

When KV cache fills up, the backend evicts old messages to make room:

1. Calculate tokens needed for new content
2. Identify message ranges to evict (oldest first, preserving system prompt)
3. Use `llama_kv_self_seq_rm()` to remove token ranges
4. Use `llama_kv_self_seq_shift()` to compact remaining tokens
5. Update session to reflect removed messages

```cpp
uint32_t evict_to_free_space(uint32_t tokens_needed);
```

## Chat Template Handling

### Template Sources

1. **llama.cpp's common_chat_templates** - Primary source, handles tool formatting
2. **minja template parser** - For token counting and custom rendering

### Tool Call Detection

The backend extracts tool call markers from the chat template:

```cpp
// Common patterns:
// - Hermes format: <tool_call> ... </tool_call>
// - Llama format: <|python_tag|> ... <|eom_id|>
```

These markers are used to detect when the model is making a tool call.

### Thinking Model Support

For models that support reasoning (Qwen3, etc.), the backend tracks thinking markers:

```cpp
std::vector<std::string> thinking_start_markers;  // e.g., "<think>"
std::vector<std::string> thinking_end_markers;    // e.g., "</think>"
```

## Generation Flow

### Sampling Chain

The sampler chain is configured in this order (per llama.cpp recommendations):

1. `top_k` - Limit vocabulary to top K tokens
2. `top_p` - Nucleus sampling
3. `penalties` - Repetition, frequency, presence penalties
4. `temperature` - Final temperature scaling
5. `dist` - Distribution sampling

```cpp
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, min_keep));
llama_sampler_chain_add(sampler, llama_sampler_init_penalties(...));
llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
```

### Cancellation Support

Generation can be cancelled via `g_generation_cancelled` global:

```cpp
// Check between batches during prompt processing
if (g_generation_cancelled) {
    LOG_INFO("Generation cancelled");
    return "";
}

// Check during token generation
if (g_generation_cancelled) {
    break;
}
```

This enables responsive Ctrl+C handling.

## Streaming Support

### add_message_stream()

The LlamaCpp backend implements `add_message_stream()` to support token-by-token streaming through the CLI server:

```cpp
Response add_message_stream(Session& session, Message::Type type,
                           const std::string& content,
                           StreamCallback callback,
                           const std::string& tool_name = "",
                           const std::string& tool_id = "",
                           int prompt_tokens = 0,
                           int max_tokens = 0) override;
```

The implementation:
1. Adds the message to the session and decodes to KV cache
2. Calls `generate(max_tokens, callback)` passing the streaming callback
3. The callback is invoked for each generated token
4. Returns the complete Response after generation finishes

### generate() with Callback

The internal `generate()` method accepts an optional callback:

```cpp
std::string generate(int max_tokens = 0, StreamCallback callback = nullptr);
```

When a callback is provided:
- Each token is decoded and passed to the callback as a delta
- The callback can return `false` to stop generation early
- Streaming output to tio is suppressed when callback is active

## Shutdown

The `shutdown()` method properly cleans up resources:

```cpp
void shutdown() {
    llama_free(model_ctx);      // Free context
    llama_model_free(model);    // Free model
    common_chat_templates_free(chat_templates);
}
```

This is called:
- When switching providers (to free GPU memory)
- In the destructor
- On application exit

## Configuration Options

From provider config JSON:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 40 | Top-K sampling limit |
| `min_keep` | int | 1 | Minimum tokens to keep |
| `penalty_repeat` | float | 1.0 | Repetition penalty (1.0 = disabled) |
| `penalty_freq` | float | 0.0 | Frequency penalty |
| `penalty_present` | float | 0.0 | Presence penalty |
| `penalty_last_n` | int | 64 | Tokens to consider for penalties |
| `gpu_layers` | int | -1 | GPU layers (-1 = auto) |
| `context_size` | int | 32768 | Context window size |
| `pipeline_parallel` / `pp` | int | 1 | Pipeline parallelism (GPUs) |
| `tensor_parallel` / `tp` | int | 1 | Tensor parallelism (GPUs) |
| `ubatch` | int | 512 | Batch size for prompt processing |

## Key Files

- `llamacpp.cpp` - Main backend implementation
- `llamacpp.h` - Header with class definition
- `llama.cpp/` - Submodule with llama.cpp library
