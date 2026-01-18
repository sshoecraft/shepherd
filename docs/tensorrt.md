# TensorRT-LLM Backend Architecture

## Overview

The TensorRT backend (`tensorrt.cpp`) provides high-performance inference using NVIDIA's TensorRT-LLM library. It supports multi-GPU inference via pipeline parallelism (PP) and integrates with MPI for cross-GPU communication.

## Multi-GPU Architecture

### Pipeline Parallelism (PP)

When a model is built with PP > 1 (e.g., PP=3 for 3 GPUs), the model layers are split across GPUs. Each GPU holds a portion of the model and they work together via MPI.

### The mpirun Re-exec Flow

TensorRT-LLM requires all GPU ranks to run the same process. When shepherd detects a multi-GPU model:

1. **Initial process** starts normally
2. Reads engine config, sees `pipeline_parallelism > 1`
3. Calls `execvp("mpirun", ...)` to replace itself with mpirun
4. **mpirun spawns N processes** (one per GPU), each running shepherd from the start
5. Each process creates its own TensorRT executor
6. The executors coordinate internally via MPI

This is why environment variable checks like `OMPI_COMM_WORLD_SIZE` matter:
- **Not set**: We're the initial process (before mpirun exec)
- **Set**: We're running under mpirun as one of the GPU ranks

### Provider Loading Message

In `factory.cpp`, we only print "Loading provider:" when `OMPI_COMM_WORLD_SIZE` is NOT set:

```cpp
if (!getenv("OMPI_COMM_WORLD_SIZE")) {
    std::cerr << "Loading provider: " << provider->name << std::endl;
}
```

This ensures the message prints exactly once (from the initial process), not N times (once per mpirun rank).

### TensorRT-LLM Log Suppression

TensorRT-LLM prints INFO logs to stdout during initialization, which interferes with model output. We suppress these by setting the log level early in initialization:

```cpp
setenv("TLLM_LOG_LEVEL", "error", 0);  // 0 = don't overwrite if already set
```

## MPI Shutdown Flow

### Architecture: mpirun with Separate Processes

Shepherd uses mpirun to spawn separate processes (one per GPU). This is different from TensorRT-LLM examples which use a single process with internal worker threads.

Our setup:
1. Initial shepherd process execs into `mpirun`
2. mpirun spawns N shepherd processes (ranks 0, 1, 2, ...)
3. Each process runs `main()` and creates its own TensorRT Executor
4. Executors coordinate internally via MPI for inference
5. **Workers wait at MPI_Barrier while rank 0 handles CLI**

### How TensorRT-LLM Handles MPI Internally

TensorRT-LLM manages MPI lifecycle:

1. **MPI_Finalize is registered as atexit handler** (`mpiUtils.cpp:186`):
   ```cpp
   std::atexit([]() { MPI_Finalize(); });
   ```

2. **Executor destructor calls shutdown()** which coordinates all ranks

### Clean Shutdown Solution

The key insight: **All ranks must exit main() together** for clean MPI shutdown.

Workers wait at a barrier while rank 0 works:

```cpp
// Workers (ranks 1, 2, ...):
if (!is_mpi_leader) {
    MPI_Barrier(MPI_COMM_WORLD);  // Wait for rank 0 to finish
    return 0;
}

// Rank 0 does CLI work...
int cli_result = run_cli(backend, session);

// Signal workers we're done
if (is_mpi) {
    MPI_Barrier(MPI_COMM_WORLD);
}

return cli_result;
// All ranks exit together, destructors run, atexit calls MPI_Finalize
```

### Why This Works

1. Workers wait at `MPI_Barrier()` after initialization
2. Rank 0 handles all CLI interaction (workers' executors participate in inference via MPI)
3. When rank 0 finishes, it hits `MPI_Barrier()`
4. All ranks synchronize and exit the barrier together
5. All ranks return from `main()` at roughly the same time
6. Destructors run on all ranks (executor cleanup)
7. `atexit` handler calls `MPI_Finalize()` on all ranks

### What Doesn't Work

- **Calling executor->shutdown() explicitly** - Hangs waiting for workers who are in our code, not the executor
- **Calling MPI_Abort** - Kills workers without cleanup, can corrupt GPU/driver state
- **Calling MPI_Finalize ourselves** - Already registered as atexit handler
- **Workers sleeping forever** - They never exit, blocking rank 0's destructor

### Error Handling

On fatal errors, `MPI_Abort()` is acceptable as last resort:

```cpp
} catch (const std::exception& e) {
    MPI_Abort(MPI_COMM_WORLD, 1);
}
```

## Chat Template Handling

### Token Context for Jinja Templates

Many jinja chat templates reference `eos_token` and `bos_token` variables. The TensorRT backend extracts these from `tokenizer_config.json` and passes them to the `ChatTemplateFactory`:

```cpp
std::string eos_tok = stop_tokens.empty() ? "" : stop_tokens[0];
chat_template_ = ChatTemplates::ChatTemplateFactory::create(
    chat_template_text_, model_config_, template_node_, eos_tok, bos_token);
```

This ensures templates like TinyLlama's (which uses `{{ message['content'] + eos_token }}`) render correctly instead of producing "None".

### Multi-Token Stop Sequences

Some models use multi-token role tags (e.g., `<|user|>` encodes to 5 tokens in TinyLlama) that aren't handled by the single-token `endId` parameter. The backend detects these patterns in the chat template and adds them as `stopWords`:

```cpp
// Detected patterns and their stop sequences:
// <|user|>, <|system|>           -> TinyLlama style
// <|start_header_id|>user/system -> Llama 3.x style
// <|im_start|>user/system        -> ChatML/Qwen style
```

These prevent the model from generating fake user/system turns by stopping generation when it starts to produce role tags.

### Thinking Models (Qwen3, etc.)

Thinking models like Qwen3 can output reasoning in `<think>` blocks. When thinking is disabled (`--nothink` or `thinking: false` in config), we inject an empty think block to signal "skip reasoning":

```
<|im_start|>assistant
<think>

</think>

```

This is handled by `Qwen3ThinkingTemplate::get_generation_prompt()` in `chat_template.cpp`.

### Chat Template Location

The chat template (`chat_template.jinja`) must be in the engine directory, not just the source model directory. When building engines, copy the template:

```bash
cp /path/to/model/chat_template.jinja /path/to/engine/chat_template.jinja
```

## Provider Switching

When switching providers via `/provider next` or `/provider <name>`:

1. **Shutdown current backend first** - frees GPU memory
2. Then create new backend

This prevents CUDA OOM errors from trying to load a second model before unloading the first.

## Streaming Token Decoding

### The Problem

BPE/SentencePiece tokenizers encode whitespace INTO the token itself (e.g., `"▁Place"` includes the leading space). When decoding tokens individually during streaming, spaces are lost:

```
Token 1: decode([1234]) -> "Place"   // Missing leading space!
Token 2: decode([5678]) -> "a"
Token 3: decode([9012]) -> "cup"
Result: "Placeacup" instead of "Place a cup"
```

### The Solution

Decode the **cumulative** token sequence and extract just the new text:

```cpp
// Track how much we've already decoded
size_t last_decoded_len = 0;

// On each new token batch:
output_tokens.insert(output_tokens.end(), beam_tokens.begin(), beam_tokens.end());
std::string full_decoded = tokenizer_->decode(output_tokens);
std::string new_text = full_decoded.substr(last_decoded_len);
last_decoded_len = full_decoded.length();
```

This ensures the tokenizer has full context to correctly place spaces.

## Two-Phase Generation (v2.21.0)

Similar to LlamaCpp, TensorRT supports two-phase generation for proper HTTP error codes:

1. **`prefill_session(session)`** - Prepares context for generation:
   - Performs prefix caching comparison with backend_session
   - Tokenizes and accumulates new messages
   - Throws `ContextFullException` if accumulated tokens exceed context_size

2. **`generate_from_prefilled(session, max_tokens)`** - Generates output:
   - Calls `generate()` with the streaming callback
   - Updates backend_session with generated assistant message

The convenience method `generate_from_session()` calls both in sequence.

### Proactive Context Check

The `generate()` method includes a proactive size check before `enqueueRequest()`:

```cpp
if (input_tokens.size() + max_tokens > context_size) {
    throw ContextFullException("Context would overflow...");
}
```

This catches context overflow before TensorRT-LLM starts processing, allowing proper error responses.

## Streaming Support

### add_message_stream()

The TensorRT backend implements `add_message_stream()` to support token-by-token streaming through the CLI server:

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
1. Tokenizes and accumulates the message
2. Calls `generate(session, max_tokens, callback)` passing the streaming callback
3. The callback is invoked for each generated token
4. Returns the complete Response after generation finishes

### generate() with Callback

The internal `generate()` method accepts an optional callback:

```cpp
std::string generate(const Session& session, int max_tokens = 0,
                    StreamCallback callback = nullptr);
```

When a callback is provided:
- Each token batch is decoded and passed to the callback
- The callback can return `false` to stop generation early
- Uses cumulative decoding to preserve whitespace (see Streaming Token Decoding section)

## Model Family Detection

The backend detects model family from `config.json` in the engine directory:

1. Reads `model_type` field (e.g., "llama", "qwen2", "chatglm")
2. For Llama models, checks for Llama 3.x tokens (`<|begin_of_text|>`, `<|eom_id|>`)
3. Falls back to Llama 2.x if no Llama 3.x tokens found

This determines chat template format, special tokens, and tool calling behavior.

## Key Files

- `tensorrt.cpp` - Main backend implementation
- `tensorrt.h` - Header with class definition
- `chat_template.cpp` - Chat template rendering with thinking model support
- `factory.cpp` - Backend creation with provider loading messages
- `models.h` - ModelFamily enum and ModelConfig factory methods
- `models.cpp` - Model detection from chat templates and paths

## Build Requirements

Requires TensorRT-LLM and MPI libraries. Enabled with `-DENABLE_TENSORRT=ON` in CMake.
