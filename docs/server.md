# Shepherd Server Architecture

## Overview

The Shepherd server provides an OpenAI-compatible HTTP API for inference using local backends (llama.cpp, TensorRT-LLM). It translates OpenAI-format requests into backend calls and converts responses back to OpenAI format.

## Components

### server.cpp

Thin wrapper that initializes and starts the API server:

```cpp
int run_server(std::unique_ptr<Backend>& backend,
               const std::string& server_host,
               int server_port);
```

### api_server.cpp

Main HTTP server implementation using cpp-httplib. Handles:

- Request parsing and validation
- Session construction from OpenAI messages
- Tool extraction and formatting
- Response conversion to OpenAI format
- Streaming via Server-Sent Events

## API Endpoints

### POST /v1/chat/completions

Main chat completion endpoint. OpenAI-compatible.

**Request:**
```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "...", "tool_call_id": "..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "tool_name",
        "description": "...",
        "parameters": {...}
      }
    }
  ],
  "stream": false,
  "max_tokens": 4096,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-name",
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

**Response with tool call:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "tool_name",
          "arguments": "{\"param\": \"value\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [{
    "id": "model-name",
    "object": "model",
    "created": 1234567890,
    "owned_by": "shepherd",
    "max_model_len": 32768
  }]
}
```

### GET /v1/models/:model_name

Get specific model info.

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "backend_connected": true
}
```

## Request Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenAI-format Request                        │
│  POST /v1/chat/completions                                       │
│  {messages: [...], tools: [...], stream: false}                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     api_server.cpp                               │
│  1. Parse JSON request                                           │
│  2. Extract tools → session.tools                                │
│  3. Extract messages → session.messages                          │
│  4. Extract system message → session.system_message              │
│  5. Parse sampling parameters                                    │
│  6. Call backend->generate_from_session(session, max_tokens)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Backend::generate_from_session()                    │
│              (llamacpp.cpp:1187)                                 │
│                                                                  │
│  1. PREFIX CACHING: Compare session with backend_session         │
│     - Find matching message prefix (already in KV cache)         │
│     - Clear diverged messages from KV cache if needed            │
│                                                                  │
│  2. SYSTEM MESSAGE FORMATTING (line 1271-1306):                  │
│     if session.system_message not empty:                         │
│       tools = session.tools or ToolRegistry                      │
│       formatted = chat_template->format_system_message(          │
│                     session.system_message, tools)               │
│       → Chat template injects tool definitions AND format        │
│       Decode formatted system message to KV cache                │
│                                                                  │
│  3. ADD NEW MESSAGES (line 1308-1347):                           │
│     For each message after matching prefix:                      │
│       format_and_decode_message() → add to KV cache              │
│                                                                  │
│  4. GENERATE RESPONSE:                                           │
│     Sample tokens until stop condition                           │
│     Return Response with content                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Response Processing                          │
│                     (api_server.cpp:502-571)                     │
│  1. Strip thinking blocks (if config->thinking == false)         │
│  2. Parse tool calls via ToolParser::parse_tool_call()           │
│  3. Convert to OpenAI response format                            │
│  4. Return JSON response                                         │
└─────────────────────────────────────────────────────────────────┘
```

## generate_from_session() - The Core Interface

This is the primary function connecting the server to the backend. Defined in `backends/backend.h`:

```cpp
virtual Response generate_from_session(
    const Session& session,
    int max_tokens = 0,
    StreamCallback callback = nullptr
) = 0;
```

### Session Structure

The Session object passed to the backend contains:

```cpp
struct Session {
    std::string system_message;           // System prompt text
    std::vector<Message> messages;        // Conversation history
    std::vector<Tool> tools;              // Available tools (from API request)

    // Sampling parameters
    float temperature, top_p, min_p;
    int top_k;
    float repetition_penalty, frequency_penalty, presence_penalty;
};
```

### Key Steps in generate_from_session (llamacpp.cpp:1187)

1. **Prefix Caching** (line 1200-1266)
   - Compares incoming `session.messages` with `backend_session` (what's in KV cache)
   - Finds longest matching prefix to avoid re-processing
   - Clears KV cache from divergence point if conversation history changed

2. **System Message Formatting** (line 1271-1306)
   ```cpp
   // Get tools from session or ToolRegistry
   std::vector<Session::Tool> tools = session.tools.empty() ?
       convert_registry_to_session_tools(ToolRegistry::instance()) : session.tools;

   // Format system message WITH TOOLS via chat template
   std::string formatted_system = chat_template->format_system_message(
       session.system_message, tools);
   ```

   **This is where tool format instructions get injected** - the chat template adds its own format specification regardless of what's in `session.system_message`.

3. **Message Processing** (line 1308-1347)
   - Adds new messages (after matching prefix) to KV cache
   - Each message formatted via `format_and_decode_message()`

4. **Token Generation**
   - Samples tokens using configured parameters
   - Calls streaming callback if provided
   - Returns Response with generated content

## Tool Call Handling

### Server-Side Flow

1. **Tools arrive** in request `tools` array (OpenAI format)
2. **Parsed** into `session.tools` vector
3. **Backend** calls `chat_template->format_system_message(system_msg, tools)`
4. **Chat template** injects tool definitions AND format instructions into system prompt
5. **Model generates** response (may include tool call in model-specific format)
6. **ToolParser** attempts to parse tool call from raw response text
7. **If found**, response converted to OpenAI `tool_calls` format

### Tool Call Detection

The server uses `ToolParser::parse_tool_call()` to detect tool calls in model output. Supported formats:

- **JSON**: `{"name": "tool", "parameters": {...}}`
- **XML wrapped JSON**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Qwen XML**: `<function=name><parameter=key>value</parameter></function>`
- **Simple XML**: `<toolname param="value"/>`

### Known Issue: Tool Format Conflicts

The chat template **always** appends its own tool format instructions to the system message:

```cpp
// chat_template.cpp - ChatMLTemplate::format_system_message()
formatted += content;  // User's system message

if (!tools.empty()) {
    formatted += "\n\n# Tools\n\n...";
    formatted += "<tool_call>\n{\"name\": ...}\n</tool_call>";  // Format instructions
}
```

This means:
- User provides system prompt with custom tool format → IGNORED
- Chat template adds its own format instructions → MODEL SEES BOTH
- Model may output unexpected format → PARSER FAILS

**Workaround**: Use `--notools` to disable tool injection, then model uses only user's system prompt.

## Streaming

When `stream: true`, the server uses Server-Sent Events (SSE):

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Thinking Block Filtering

During streaming, if `config->thinking == false`, thinking blocks are filtered:

1. Buffer incoming tokens
2. Detect `<think>` start marker → begin suppressing
3. Detect `</think>` end marker → resume output
4. Only non-thinking content sent to client

## Concurrency

The server uses a **single mutex** to serialize all backend requests:

```cpp
static std::mutex backend_mutex;

svr.Post("/v1/chat/completions", [&](...) {
    std::lock_guard<std::mutex> lock(backend_mutex);
    // ... handle request
});
```

This means:
- One request processed at a time
- Thread-safe for single backend instance
- No concurrent inference (backend limitation)

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": "400"
  }
}
```

### Error Types

| Status | Type | When |
|--------|------|------|
| 400 | `invalid_request_error` | Invalid JSON, context limit exceeded |
| 401 | `authentication_error` | Auth failure (not used currently) |
| 429 | `rate_limit_error` | Rate limiting (not used currently) |
| 500 | `server_error` | Backend errors, exceptions |

### Context Limit Detection

The server detects context limit errors and returns 400 instead of 500:

```cpp
static bool is_context_limit_error(const std::string& error_msg) {
    // Checks for "context limit", "context window", etc.
}
```

## Configuration

The server inherits configuration from the global `config` object:

| Setting | Effect |
|---------|--------|
| `config->thinking` | If false, strip thinking blocks from responses |

## Key Files

- `server/server.cpp` - Entry point, calls api_server
- `server/server.h` - Server interface
- `server/api_server.cpp` - HTTP server implementation
- `server/api_server.h` - API server interface
- `tools/tool_parser.cpp` - Tool call parsing
- `backends/chat_template.cpp` - System prompt + tool formatting
