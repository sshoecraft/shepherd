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

## Version History

- **2.13.0** - Unified EventCallback pattern (no add_message_stream)
- **2.6.0** - Added APIServer class wrapper
- **2.5.0** - Original implementation via run_api_server()
