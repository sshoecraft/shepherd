# API Server Module Documentation

## Overview

The API Server provides an OpenAI-compatible HTTP API for inference using local backends (llama.cpp, TensorRT-LLM). It translates OpenAI-format requests into backend calls and converts responses back to OpenAI format.

## Architecture

### Files

- `frontends/api_server.h` - Class declaration
- `frontends/api_server.cpp` - HTTP server implementation

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
shepherd --server --port 8000
```

With a specific provider:

```bash
shepherd --server --port 8000 --provider mylocal
```

## Authentication

By default, the API server requires no authentication. For remote access, enable API key authentication:

```bash
# Start server with authentication
./shepherd --server --auth-mode json

# Generate an API key
shepherd apikey add mykey
# Output: sk-shep-abc123...

# Use the key in requests
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-shep-abc123..." \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

**API Key Management:**
```bash
shepherd apikey add <name>      # Generate new key
shepherd apikey list            # List all keys
shepherd apikey remove <name>   # Remove a key
```

## Server-Side Tools

By default, the API server does **not** expose Shepherd's tools via HTTP endpoints. Tools are provided by the client in each request and executed client-side.

To enable server-side tool access (for MCP proxy use cases):

```bash
./shepherd --server --server-tools
```

When `--server-tools` is enabled, two additional endpoints become available:

- `GET /v1/tools` - List all available tools
- `POST /v1/tools/execute` - Execute a tool by name

**Security Warning**: Only enable `--server-tools` when you need remote tool execution. These endpoints expose shell commands, file system access, and MCP server tools to anyone who can reach the API.

## Session Management

The API server follows standard OpenAI protocol - each request contains the full conversation history, and the server processes it accordingly.

### With Local Backends (llamacpp, TensorRT)

When using local backends, Shepherd maintains a KV cache for efficient multi-turn conversations:

1. **Message Comparison**: Each request sends the full conversation history (OpenAI protocol)
2. **Prefix Matching**: Server compares incoming messages with cached tokens
3. **KV Cache Reuse**: Matching prefix is skipped; only new tokens are processed
4. **Optimization**: Subsequent turns in the same conversation are much faster

### With API Backends (OpenAI, Anthropic, etc.)

With cloud API backends, each request is forwarded to the upstream provider - fully stateless on the Shepherd side.

### Prefix Caching (vLLM-style)

When a new request arrives, the server:

1. **Compares** incoming messages with cached tokens
2. **Detects** the longest matching prefix
3. **Reuses** KV cache for matching tokens
4. **Only processes** new tokens after the prefix

**Example:**

```
Request 1: [system, user_1]
  → Server processes: system (4141 tokens) + user_1 (10 tokens)
  → KV cache: 4151 tokens
  → Time: ~1.5s (full processing)

Request 2: [system, user_1, assistant_1, user_2]
  → Server detects: system + user_1 match cached (4151 tokens)
  → Server processes ONLY: assistant_1 (27 tokens) + user_2 (8 tokens)
  → KV cache: 4151 + 35 = 4186 tokens
  → Time: ~200ms (90% faster!)
```

**Performance Comparison:**

| Scenario | Without Prefix Caching | With Prefix Caching | Speedup |
|----------|------------------------|---------------------|---------|
| Turn 1 (4151 tokens) | 1.5s | 1.5s | 1x |
| Turn 2 (35 new tokens) | 1.6s | 0.2s | **8x** |
| Turn 3 (117 new tokens) | 1.7s | 0.18s | **9x** |
| Turn 10 (50 new tokens) | 2.1s | 0.15s | **14x** |

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

**Response:**
```json
{
  "status": "healthy",
  "backend": "llamacpp",
  "model": "/models/qwen-3-30b.gguf"
}
```

### GET /v1/tools (requires `--server-tools`)

List available tools for remote execution.

**Response:**
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "Path to the file"}
        },
        "required": ["path"]
      }
    }
  ]
}
```

### POST /v1/tools/execute (requires `--server-tools`)

Execute a tool by name.

**Request:**
```json
{
  "name": "read_file",
  "tool_call_id": "call_123",
  "arguments": {"path": "/tmp/example.txt"}
}
```

**Response:**
```json
{
  "tool_call_id": "call_123",
  "success": true,
  "content": "File contents here..."
}
```

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

## Client Examples

### Python (OpenAI Library)

```python
import openai

client = openai.OpenAI(
    api_key="sk-shep-abc123...",  # Or "dummy" if --auth-mode none
    base_url="http://localhost:8000/v1"
)

# Multi-turn conversation
messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})

    print(f"Assistant: {assistant_msg}")
```

### curl

```bash
# Single request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# With tools
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "List files in /tmp"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "list_directory",
          "description": "List files",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string"}
            }
          }
        }
      }
    ]
  }'
```

## Important Notes

- **Tools**: In server mode, tools are provided by the **client** in each request
  - Server does NOT execute tools by default (client-side execution)
  - Server returns tool calls to client for execution
  - Client executes tools and sends results back in next request
  - Use `--server-tools` to enable server-side tool execution

- **KV Cache (Local Backends)**: With llamacpp/TensorRT, the KV cache persists across requests
  - Prefix matching provides fast multi-turn conversations
  - Clear cache by restarting the server

## Version History

- **2.21.1** - Fix non-streaming responses dropping CODEBLOCK events (content inside markdown code blocks was not accumulated)
- **2.17.0** - Two-phase streaming for proper HTTP error codes
- **2.13.0** - Unified EventCallback pattern (no add_message_stream)
- **2.6.0** - Added APIServer class wrapper
- **2.5.0** - Original implementation via run_api_server()
