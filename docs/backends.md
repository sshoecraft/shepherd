# Backends Module

## Overview

The backends module (`backends/`) contains all LLM backend implementations. Each backend implements the abstract `Backend` class and provides token generation via the EventCallback pattern.

## Backend Base Class

```cpp
// backends/backend.h

class Backend {
public:
    // Event callback for streaming output
    using EventCallback = std::function<bool(
        Message::Type type,
        const std::string& content,
        const std::string& tool_name,
        const std::string& tool_call_id
    )>;

    // Main interface - add message and generate response
    virtual Response add_message(
        Session& session,
        Message::Type type,
        const std::string& content,
        EventCallback callback = nullptr,  // nullptr = non-streaming
        const std::string& tool_name = "",
        const std::string& tool_id = "",
        int prompt_tokens = 0,
        int max_tokens = 0
    ) = 0;

    // Stateless generation (for servers with prefix caching)
    virtual Response generate_from_session(
        const Session& session,
        int max_tokens = 0,
        EventCallback callback = nullptr
    ) = 0;
};
```

## EventCallback Details

### Signature

```cpp
bool callback(
    Message::Type type,           // Type of event
    const std::string& content,   // Event content
    const std::string& tool_name, // Tool name (if applicable)
    const std::string& tool_call_id // Tool ID (if applicable)
);
```

### Event Types

| Message::Type | content | tool_name | tool_call_id | Description |
|---------------|---------|-----------|--------------|-------------|
| ASSISTANT | delta text | "" | "" | Text token generated |
| TOOL_REQUEST | tool args JSON | tool name | call ID | Tool call detected |
| TOOL_RESPONSE | result | tool name | call ID | Tool result (input event) |

### Return Value

- `true`: Continue generation
- `false`: Cancel generation immediately

### Invocation Pattern

```cpp
// Inside backend's add_message():
for (each generated token) {
    if (callback) {
        if (!callback(Message::ASSISTANT, token_text, "", "")) {
            break;  // User cancelled
        }
    }
    accumulated += token_text;
}
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
if (!callback(Message::ASSISTANT, delta_text, "", "")) {
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
std::string generate(int max_tokens, EventCallback callback) {
    while (tokens_generated < max_tokens) {
        // Sample next token
        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        // Decode to text
        std::string text = token_to_text(token);

        // Invoke callback
        if (callback) {
            if (!callback(Message::ASSISTANT, text, "", "")) {
                break;
            }
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

Key features:
- HTTP client to CLI server
- SSE listener for streaming
- Tool execution on server side

Note: This backend is special - it receives events FROM a remote server rather than generating them locally. The SSE listener thread handles incoming events.

## Implementing a New Backend

1. Inherit from `Backend` (or `ApiBackend` for HTTP APIs)
2. Implement `add_message()`:
   ```cpp
   Response MyBackend::add_message(Session& session,
                                   Message::Type type,
                                   const std::string& content,
                                   EventCallback callback,
                                   ...) {
       // If no callback, use non-streaming
       if (!callback) {
           return non_streaming_implementation(...);
       }

       // Streaming implementation
       std::string accumulated;
       while (generating) {
           std::string token = generate_next_token();

           // CRITICAL: Invoke callback for each token
           if (!callback(Message::ASSISTANT, token, "", "")) {
               break;  // Cancelled
           }

           accumulated += token;
       }

       Response resp;
       resp.content = accumulated;
       resp.success = true;
       return resp;
   }
   ```

3. Implement `generate_from_session()` for stateless generation
4. Register in `backends/factory.cpp`

## Response Structure

```cpp
struct Response {
    bool success;
    Code code;  // SUCCESS, ERROR, RATE_LIMITED, etc.
    std::string content;
    std::string error;
    std::string finish_reason;  // "stop", "length", "tool_calls", "cancelled"
    int prompt_tokens;
    int completion_tokens;
    std::vector<ToolParser::ToolCall> tool_calls;
    bool was_streamed;
};
```

## Tool Call Handling

Backends detect tool calls in two ways:

1. **API-native**: Response includes `tool_calls` field (OpenAI, Anthropic)
2. **Content-based**: Parse tool call markers from response text (local models)

The backend populates `Response::tool_calls`, frontend handles execution.

```cpp
// After generation:
if (!resp.tool_calls.empty()) {
    resp.finish_reason = "tool_calls";
}
```

## Error Handling

Backends should:
- Set `resp.success = false` on errors
- Populate `resp.error` with description
- Set appropriate `resp.code`
- Return partial content if available

```cpp
if (http_error) {
    Response resp;
    resp.success = false;
    resp.code = Response::ERROR;
    resp.error = "HTTP request failed: " + error_msg;
    return resp;
}
```
