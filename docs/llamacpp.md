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

### Two-Phase Generation (v2.21.0)

Generation can be split into two phases to support proper HTTP error codes in streaming mode:

1. **`prefill_session(session)`** - Prepares context for generation:
   - Renders conversation via chat template
   - Tokenizes the rendered text
   - Compares with KV cache mirror for prefix caching
   - Decodes delta tokens into KV cache
   - Throws `ContextFullException` if context would overflow

2. **`generate_from_prefilled(session, max_tokens)`** - Generates output:
   - Calls `generate()` with the streaming callback
   - Updates session token counts
   - Fires STOP callback

The convenience method `generate_from_session()` simply calls both in sequence:

```cpp
void generate_from_session(Session& session, int max_tokens) {
    prefill_session(session);
    generate_from_prefilled(session, max_tokens);
}
```

This split allows the API server to catch `ContextFullException` BEFORE committing to streaming (before HTTP 200 is sent), returning a proper HTTP 400 error response.

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

## Tool Call Handling

Tool calls are detected during streaming via `Backend::output()` filter:

1. All generated content flows through `output()` in the base Backend class
2. `output()` detects tool call markers (JSON `{...}` or XML `<tool_call>`)
3. `emit_tool_call()` parses the tool call (JSON or XML format) and stores in `pending_tool_calls`
4. After generation, `add_message()` builds `tool_calls_json` from `pending_tool_calls`
5. The assistant message is stored with `tool_calls_json` populated

```cpp
// After generate() returns:
if (!pending_tool_calls.empty()) {
    nlohmann::json tc_array = nlohmann::json::array();
    for (const auto& tc : pending_tool_calls) {
        // Build OpenAI-format tool call JSON
        tc_array.push_back({...});
    }
    assistant_msg.tool_calls_json = tc_array.dump();
}
```

This ensures proper Jinja template formatting when TOOL_RESPONSE messages are added.

### Backend Session Synchronization

After generation completes, the assistant message must be added to `backend_session.messages` to keep it synchronized with the KV cache:

```cpp
// After generation - always add if tokens were generated
if (last_assistant_kv_tokens > 0) {
    Message assistant_msg(Message::ASSISTANT, result, last_assistant_kv_tokens);
    backend_session.messages.push_back(assistant_msg);
    backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
}
```

This is critical for prefix caching in server mode. The condition checks `last_assistant_kv_tokens > 0` rather than `!result.empty()` to handle channel-based models (like GPT-OSS) where content extraction may return empty if the model doesn't reach the final channel (e.g., due to max_tokens limits). Without proper sync:
- The KV cache contains stray tokens (generation_prompt + partial_content + closing_tag)
- Next request's prefix matching doesn't detect divergence (backend_session doesn't have these tokens)
- Next generation continues from stale KV cache state, causing alternating success/failure patterns

## O(1) Message Formatting Optimization

The `format_and_decode_message()` function uses an O(1) approach for formatting messages, avoiding the O(n²) overhead of rendering the entire conversation for each new message.

### Message Type Handling

| Message Type | Strategy | Complexity |
|-------------|----------|------------|
| SYSTEM | Use pre-formatted content directly | O(1) |
| USER | `format_message(msg)` single-message render | O(1) |
| ASSISTANT (simple) | Use raw content directly | O(1) |
| ASSISTANT (tool_calls) | 2-element vector `[prev, current]` | O(1) |
| TOOL_RESPONSE | 2-element vector `[prev, current]` | O(1) |
| Harmony/Channels | 2-element vector `[prev, current]` | O(1) |

### 2-Element Vector Technique

For messages requiring template state (e.g., `loop.previtem` in Qwen, `ns.is_tool` in DeepSeek), we pass a 2-element vector containing just the previous message and current message:

```cpp
if (target_index > 0) {
    std::vector<Message> two_msgs = {all_messages[target_index - 1], msg};
    rendered_msg = chat_template->format_message_incremental(two_msgs, 1, tools, false);
}
```

This works because:
- Template processes prev message first (sets state variables)
- Template processes current message with correct state
- `format_message_incremental` extracts just the current message's output
- All template state dependencies only need the PREVIOUS message

### Templates Verified

- **Qwen3-Coder**: `loop.previtem` for tool responses ✓
- **DeepSeek-R1**: `ns.is_tool`, `ns.is_output_first` state vars ✓
- **TinyLlama**: Simple wrapping ✓
- **Qwen2.5**: `loop.first` for system ✓
- **GPT-OSS (Harmony)**: Channel wrapping ✓

## Special Token Rendering

When extracting EOS/BOS tokens from the vocabulary for template rendering, the `special` parameter to `llama_token_to_piece()` must be `true`. This ensures special tokens like `<｜begin▁of▁sentence｜>` are rendered as text strings rather than empty or garbage output.

```cpp
// Correct - renders special tokens as text
int len = llama_token_to_piece(vocab, bos_id, buf, sizeof(buf), 0, true);

// Wrong - may return empty or garbage for special tokens
int len = llama_token_to_piece(vocab, bos_id, buf, sizeof(buf), 0, false);
```

This is critical for models like DeepSeek-R1 that use fullwidth Unicode characters in their special tokens.

## Template Rendering Fallback

The `render_message()` function wraps template rendering in a try-catch to handle templates with unsupported features:

```cpp
try {
    return chat_template->format_message_incremental(...);
} catch (const std::exception& e) {
    // Fall back to simple ChatML format
    return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
}
```

This prevents crashes when templates use features not fully supported by minja (e.g., certain iteration patterns over tool call arguments).

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
| `n_batch` | int | 512 | Logical batch size for prompt processing |
| `ubatch` / `n_ubatch` | int | 512 | Physical micro-batch size (must be ≤ n_batch) |

## Harmony Parser Integration (GPT-OSS Models)

For models using OpenAI's Harmony format (channel-based output), the backend integrates a Rust-based token parser via FFI.

### Parser Initialization

The Rust parser is created at the start of `run_inference()` and needs to be pre-seeded with the generation prompt tokens:

```cpp
// Pre-feed generation prompt tokens to parser
// The generation prompt (e.g., "<|start|>assistant") is already in KV cache
// but the parser needs to see these tokens to be in the correct state
for (llama_token token : generation_prompt_tokens) {
    rust_harmony_parser->process(static_cast<uint32_t>(token), nullptr);
}
```

This is critical because:
1. The generation prompt (`<|start|>assistant`) is decoded into KV cache before generation
2. The model continues from that point, generating channel markers like `<|channel|>`
3. The parser expects to see `<|start|>` first to transition from ExpectStart state
4. Without pre-seeding, the parser errors on the first generated token

### Stop Token Handling

The parser uses different stop tokens for different purposes:
- **Generation stopping** (C FFI): `<|return|>` and `<|call|>` only
- **Internal message parsing**: `<|return|>`, `<|call|>`, and `<|end|>`

This allows `<|end|>` to end a channel without stopping generation, enabling multi-channel responses.

## Key Files

- `llamacpp.cpp` - Main backend implementation
- `llamacpp.h` - Header with class definition
- `llama.cpp/` - Submodule with llama.cpp library
- `harmony_rust.cpp/h` - C++ wrapper for Rust FFI
