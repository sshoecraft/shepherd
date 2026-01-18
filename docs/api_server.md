# API Server Module Documentation

## Overview

The API Server provides an OpenAI-compatible HTTP API for inference using local backends (llama.cpp, TensorRT-LLM). It translates OpenAI-format requests into backend calls and converts responses back to OpenAI format.

## Architecture

### Files

- `server/api_server.h` - Class declaration
- `server/api_server.cpp` - HTTP server implementation

### Class Structure

```cpp
class APIServer : public Server {
public:
    APIServer(const std::string& host, int port);
    ~APIServer();

    int run(Session& session) override;
};
```

## Usage

Start API server mode:

```bash
shepherd --apiserver --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

**Request:**
```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "tools": [...],
  "stream": false,
  "max_tokens": 4096,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Response text"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

### GET /v1/models

List available models.

### GET /health

Health check endpoint.

## Streaming (v2.13.0)

When `stream: true`, the server uses EventCallback:

```cpp
auto stream_callback = [&](Message::Type type,
                           const std::string& content,
                           const std::string& tool_name,
                           const std::string& tool_call_id) -> bool {
    // Format as OpenAI chunk
    nlohmann::json chunk = {
        {"choices", {{
            {"delta", {{"content", content}}}
        }}}
    };
    send_sse("data: " + chunk.dump() + "\n\n");
    return true;
};

Response resp = backend->generate_from_session(session, max_tokens, stream_callback);
```

SSE Response format:
```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Tool Call Handling

1. Tools arrive in request `tools` array
2. Backend generates response (may include tool call)
3. ToolParser extracts tool calls from response
4. Response converted to OpenAI `tool_calls` format
5. Client executes tools and sends results in next request

## Concurrency

Single mutex serializes all backend requests. One request processed at a time.

## Error Responses

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": "400"
  }
}
```

## Two-Phase Streaming (v2.21.0)

Prior to v2.21.0, streaming mode would send HTTP 200 headers before the backend started processing. If context overflow was detected during prefill, the error could only be sent as an SSE event - the HTTP status was already committed.

Now streaming uses a two-phase approach:

1. **Prefill Phase** (before HTTP headers sent):
   - Call `backend->prefill_session(session)`
   - If `ContextFullException` thrown → return HTTP 400 with JSON error
   - This happens BEFORE `set_content_provider()` commits to streaming

2. **Generate Phase** (after 200 committed):
   - Call `backend->generate_from_prefilled(session, max_tokens)` inside content provider
   - Tokens stream normally via SSE events

This ensures proper HTTP semantics: context overflow returns HTTP 400, not 200 + SSE error.

**Note:** This only applies to local backends (LlamaCpp, TensorRT). API backends (OpenAI, Anthropic, etc.) use default no-op implementations since they can't validate against the remote server's context limit.

## Version History

- **2.21.1** - Fix non-streaming responses dropping CODEBLOCK events (content inside markdown code blocks was not accumulated)
- **2.17.0** - Two-phase streaming for proper HTTP error codes
- **2.13.0** - Unified EventCallback pattern (no add_message_stream)
- **2.6.0** - Added APIServer class wrapper
- **2.5.0** - Original implementation via run_api_server()
