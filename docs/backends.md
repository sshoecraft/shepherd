# Backends Module

## Overview

The backends module (`backends/`) contains all LLM backend implementations. Each backend implements the abstract `Backend` class and provides token generation via the EventCallback pattern.

## Backend Base Class

```cpp
// backend.h

class Backend {
public:
    // Event callback for streaming output
    using EventCallback = std::function<bool(
        CallbackEvent event,           // Type of event
        const std::string& content,    // Event content
        const std::string& name,       // Tool name (for TOOL_CALL)
        const std::string& id          // Tool call ID
    )>;

    // Main interface - add message and generate response
    virtual void add_message(
        Session& session,
        Message::Role role,
        const std::string& content,
        const std::string& tool_name = "",
        const std::string& tool_id = "",
        int max_tokens = 0
    ) = 0;

    // Stateless generation (for servers with prefix caching)
    virtual void generate_from_session(Session& session, int max_tokens = 0) = 0;

    // Two-phase generation for streaming with proper HTTP error codes (v2.21.0)
    // Local backends: Split prefill (decode to KV cache) from generate (inference)
    // API backends: No-op / passthrough (can't validate against remote server)
    virtual void prefill_session(Session& session) {}  // May throw ContextFullException
    virtual void generate_from_prefilled(Session& session, int max_tokens = 0) {
        generate_from_session(session, max_tokens);  // Default: full operation
    }

    // Token counting for eviction decisions
    virtual int count_message_tokens(
        Message::Role role,
        const std::string& content,
        const std::string& tool_name = "",
        const std::string& tool_id = ""
    ) = 0;

    // Public members
    EventCallback callback;       // Set by frontend before calling add_message
    std::string model_name;
    size_t context_size;
    int max_output_tokens;
    bool is_local;               // true for GPU backends, false for API
    bool streaming_enabled;
    bool sse_handles_output;     // true if SSE handles display (CLI client)
};
```

## CallbackEvent Types

```cpp
enum class CallbackEvent {
    CONTENT,      // Assistant text chunk
    THINKING,     // Reasoning/thinking chunk (if show_thinking enabled)
    TOOL_CALL,    // Model requesting a tool call
    TOOL_RESULT,  // Result of tool execution
    USER_PROMPT,  // Echo user's prompt
    SYSTEM,       // System info/status messages
    ERROR,        // Error occurred
    STOP,         // Generation complete (finish_reason in content)
    CODEBLOCK,    // Code block content (inside ```)
    STATS         // Performance stats (prefill/decode speed)
};
```

## EventCallback Details

### Signature

```cpp
bool callback(
    CallbackEvent event,          // Type of event
    const std::string& content,   // Event content
    const std::string& name,      // Tool name (if applicable)
    const std::string& id         // Tool call ID (if applicable)
);
```

### Event Parameters

| CallbackEvent | content | name | id | Description |
|---------------|---------|------|-----|-------------|
| CONTENT | delta text | "" | "" | Text token generated |
| THINKING | thinking text | "" | "" | Reasoning content |
| TOOL_CALL | tool args JSON | tool name | call ID | Tool call detected |
| TOOL_RESULT | result summary | tool name | call ID | Tool result |
| USER_PROMPT | user input | "" | "" | Echo of user prompt |
| SYSTEM | message | "" | "" | System notification |
| ERROR | error message | error type | "" | Error occurred |
| STOP | finish_reason | "" | "" | Generation complete |
| CODEBLOCK | code content | "" | "" | Code block content |
| STATS | stats JSON | "" | "" | Performance metrics |

### Return Value

- `true`: Continue generation
- `false`: Cancel generation immediately

### Invocation Pattern

```cpp
// Inside backend's add_message():
for (each generated token) {
    if (!callback(CallbackEvent::CONTENT, token_text, "", "")) {
        break;  // User cancelled
    }
    accumulated += token_text;
}

// At end of generation:
callback(CallbackEvent::STOP, finish_reason, "", "");
```

## Backend Implementations

### ApiBackend (api.cpp)

Base class for HTTP API backends. Provides:
- HTTP client integration
- Token counting via EMA calibration
- Response parsing framework

```cpp
class ApiBackend : public Backend {
    // Subclasses implement these:
    virtual nlohmann::json build_request(...) = 0;
    virtual Response parse_http_response(...) = 0;
    virtual std::map<std::string, std::string> get_api_headers() = 0;
    virtual std::string get_api_endpoint() = 0;
};
```

### OpenAIBackend (openai.cpp)

OpenAI API and compatible endpoints (vLLM, local-ai, etc).

Key features:
- Chat completions API
- Streaming via SSE
- Tool/function calling support
- Vision (image) support

Callback invocation:
```cpp
// In SSE stream handler:
if (!callback(CallbackEvent::CONTENT, delta_text, "", "")) {
    return false;  // Stop streaming
}
```

### AnthropicBackend (anthropic.cpp)

Claude API implementation.

Key features:
- Messages API
- Streaming via SSE
- Tool use support
- System prompt handling

### GeminiBackend (gemini.cpp)

Google Gemini API implementation.

Key features:
- generateContent API
- Streaming support
- Function calling

### OllamaBackend (ollama.cpp)

Local Ollama server.

Key features:
- Chat API
- Streaming via NDJSON
- Local model management

### LlamaCppBackend (llamacpp.cpp)

Direct llama.cpp integration (no server).

Key features:
- KV cache management
- Token-by-token generation
- Chat template formatting
- Tool call marker detection

Internal generate function:
```cpp
std::string generate(int max_tokens) {
    while (tokens_generated < max_tokens) {
        // Sample next token
        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        // Decode to text
        std::string text = token_to_text(token);

        // Invoke callback via process_output (handles filtering)
        if (!process_output(text)) {
            break;
        }

        response += text;
    }
    return response;
}
```

### TensorRTBackend (tensorrt.cpp)

NVIDIA TensorRT-LLM backend.

Key features:
- High-performance inference
- Multi-GPU support
- KV cache management
- Batch scheduling

### CLIClientBackend (cli_client.cpp)

Proxy to remote CLI server.

**See [docs/cliserver.md](cliserver.md) for complete documentation.**

Key features:
- HTTP client to CLI server
- SSE listener for streaming
- Tool execution on server side

This backend is unique because:
- It receives events FROM a remote server rather than generating locally
- SSE listener thread handles incoming broadcasts
- The `sse_handles_output = true` flag indicates display is handled by SSE
- It only processes USER messages (server manages full conversation)

## Implementing a New Backend

1. Inherit from `Backend` (or `ApiBackend` for HTTP APIs)
2. Implement `add_message()`:
   ```cpp
   void MyBackend::add_message(Session& session,
                               Message::Role role,
                               const std::string& content,
                               const std::string& tool_name,
                               const std::string& tool_id,
                               int max_tokens) {
       // Reset output state for new generation
       reset_output_state();

       // Streaming implementation
       std::string accumulated;
       while (generating) {
           std::string token = generate_next_token();

           // Route through output filter (handles tool calls, thinking, etc.)
           if (!process_output(token)) {
               break;  // Cancelled
           }

           accumulated += token;
       }

       // Flush any buffered output
       flush_output();

       // Signal completion
       callback(CallbackEvent::STOP, "stop", "", "");

       // Update session with assistant message
       Message msg(Message::ASSISTANT, accumulated);
       msg.tokens = completion_tokens;
       session.messages.push_back(msg);
   }
   ```

3. Implement `generate_from_session()` for stateless generation
4. Implement `count_message_tokens()` for eviction calculations
5. Register in provider connection logic

## Output Filtering

The `Backend` base class provides `process_output()` which:
- Routes through channel parser (for GPT-OSS harmony format)
- Detects tool call markers and buffers them
- Detects thinking blocks and routes to THINKING callback
- Tracks code blocks (```) for CODEBLOCK events
- Batches small outputs for efficiency

```cpp
// Always use process_output() instead of callback directly:
bool continue_gen = process_output(generated_text);
```

## Response Structure

```cpp
struct Response {
    enum Code {
        SUCCESS = 0,
        ERROR = 1,
        CONTEXT_FULL = 2,
        MAX_TOKENS_TOO_HIGH = 3
    };

    Code code;
    bool success;
    std::string content;
    std::string error;
    std::string finish_reason;  // "stop", "length", "tool_calls"
    int prompt_tokens;
    int completion_tokens;
    std::vector<ToolParser::ToolCall> tool_calls;
    std::string tool_calls_json;
    bool was_streamed;
};
```

## Tool Call Handling

Backends detect tool calls in two ways:

1. **API-native**: Response includes `tool_calls` field (OpenAI, Anthropic, Gemini)
   - Parsed from structured JSON in API response
   - Stored in `resp.tool_calls` and `resp.tool_calls_json`

2. **Content-based**: Parse tool call markers from response text (local models, vLLM)
   - All content routes through `Backend::output()` filter
   - `output()` detects JSON tool calls (`{...}`) or XML markers (`<tool_call>`)
   - `emit_tool_call()` parses and stores in `Backend::pending_tool_calls`
   - After generation, `pending_tool_calls` populates `assistant_msg.tool_calls_json`

### Tool Call Parsing Formats

`emit_tool_call()` in `backend.cpp` supports multiple parsing formats:

1. **JSON format** (Hermes, OpenAI-compatible):
   ```json
   {"name": "tool_name", "arguments": {"key": "value"}}
   ```

2. **XML format** (Qwen3-Coder style):
   ```xml
   <tool_call>
   <function=tool_name>
   <arg_name>arg_value</arg_name>
   </function>
   </tool_call>
   ```

The parser first attempts JSON parsing. If no tool name is found, it falls back to XML parsing for `<function=name>` format.

Both paths result in `tool_calls_json` being set on the assistant message, which is required for proper Jinja template formatting of subsequent TOOL_RESPONSE messages.

```cpp
// API backends: from structured response
assistant_msg.tool_calls_json = resp.tool_calls_json;

// Local backends: from output() filter detection
if (!pending_tool_calls.empty()) {
    // Build OpenAI-format JSON array
    assistant_msg.tool_calls_json = build_tool_calls_json(pending_tool_calls);
}
```

## Error Handling

Backends should:
- Signal non-context errors via callback: `callback(CallbackEvent::ERROR, message, type, "")`
- Throw `ContextFullException` for context-full errors (see below)
- Return partial content if available

### Context Full: ContextFullException (v2.33.1)

**Only the session owner (frontend) performs evictions.** Backends never evict.

When a backend detects a context-full condition, it throws `ContextFullException`.
The frontend catches it in `generate_response()` and handles reactive eviction:

```cpp
// Backend detects context full:
int tokens_to_evict = extract_tokens_to_evict(http_response);
if (tokens_to_evict > 0) {
    throw ContextFullException(error_msg);
}

// Frontend catches and evicts (frontend.cpp):
catch (const ContextFullException& e) {
    auto ranges = session.calculate_messages_to_evict(tokens_over);
    session.evict_messages(ranges);
    // retry generation
}
```

This pattern is used by all backends:
- **Local backends** (llamacpp, tensorrt): Throw during prefill/decode when KV cache is full
- **API backends** (OpenAI, Anthropic, Gemini, Ollama): Parse HTTP error response
  via `extract_tokens_to_evict()`, throw if context-full detected

For non-context errors:
```cpp
callback(CallbackEvent::ERROR, "HTTP request failed: " + error_msg, "http_error", "");
callback(CallbackEvent::STOP, "error", "", "");
```

## Files

- `backend.h` / `backend.cpp` - Base Backend class with output filtering
- `backends/api.h` / `backends/api.cpp` - API backend base class
- `backends/openai.cpp` - OpenAI/compatible backend
- `backends/anthropic.cpp` - Anthropic Claude backend
- `backends/gemini.cpp` - Google Gemini backend
- `backends/ollama.cpp` - Ollama backend
- `backends/llamacpp.cpp` - Direct llama.cpp backend
- `backends/tensorrt.cpp` - TensorRT-LLM backend
- `backends/cli_client.cpp` - CLI server proxy backend
