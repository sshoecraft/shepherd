# CLI Server Module Documentation

## Overview

The CLI Server provides an HTTP server mode that executes tools locally while accepting prompts from remote clients. This enables a local machine with tools (filesystem access, shell commands) to serve as a backend for remote interfaces.

## Architecture

### Files

- `server/cli_server.h` - Class declaration
- `server/cli_server.cpp` - Implementation
- `backends/cli_client.h` - Client backend declaration
- `backends/cli_client.cpp` - Client backend implementation

### Class Structure

```cpp
class CLIServer : public Server {
public:
    CLIServer(const std::string& host, int port);
    ~CLIServer();

    void init(Session& session, bool no_mcp, bool no_tools) override;
    int run(Session& session) override;
};
```

## Usage

### Server Mode

Start CLI server mode:

```bash
shepherd --cliserver --host 0.0.0.0 --port 8000
```

### Client Mode

Connect to a CLI server using the `cli` provider:

```bash
shepherd --provider cli
```

Configure the CLI provider in `providers.json`:

```json
{
    "cli": {
        "type": "cli",
        "api_base": "http://localhost:8000",
        "model": "remote-model"
    }
}
```

## API Endpoints

### POST /request

Main request endpoint for prompts.

**Request:**
```json
{
    "prompt": "user message",
    "stream": true
}
```

**Parameters:**
- `prompt` - The user message to process
- `stream` - Enable SSE streaming (default: false)

**Response (streaming):**

Server-Sent Events (SSE) format:
```
data: {"type": "delta", "content": "Hello"}

data: {"type": "delta", "content": " world"}

data: {"type": "tool_call", "name": "read_file", "args": {...}}

data: {"type": "tool_result", "success": true, "content": "..."}

data: {"type": "response_complete", "finish_reason": "stop"}
```

**Response (non-streaming):**
```json
{
    "response": "Hello world",
    "success": true
}
```

### GET /updates

SSE endpoint for real-time event streaming. Clients connect here to receive events broadcast by the server.

### GET /status

Server status and session info.

## Streaming (v2.13.0)

The CLI server uses EventCallback for streaming:

```cpp
auto stream_callback = [&](Message::Type type,
                           const std::string& content,
                           const std::string& tool_name,
                           const std::string& tool_call_id) -> bool {
    if (content.empty()) return true;

    // Filter tool calls from stream
    pending_buffer += content;
    size_t marker_pos = find_earliest_marker(pending_buffer, markers);

    if (marker_pos != std::string::npos) {
        // Output up to marker, buffer the rest
        std::string output = pending_buffer.substr(0, marker_pos);
        pending_buffer = pending_buffer.substr(marker_pos);
        if (!output.empty()) {
            broadcast_sse("delta", {{"content", output}});
        }
    } else {
        broadcast_sse("delta", {{"content", pending_buffer}});
        pending_buffer.clear();
    }
    return true;
};

// Unified add_message with callback
Response resp = session->add_message(Message::USER, prompt, stream_callback);
```

## Backend Support

All backends support EventCallback streaming:

| Backend | Streaming Support |
|---------|------------------|
| LlamaCpp | Yes (callback per token) |
| TensorRT | Yes (callback per token) |
| Anthropic | Yes (SSE parsing) |
| OpenAI | Yes (SSE parsing) |
| Gemini | Yes (SSE parsing) |
| Ollama | Yes (NDJSON parsing) |

Backends without streaming invoke callback once with full response.

## Tool Execution Flow

```
Client                   CLIServer                Backend
  │                         │                        │
  ├──POST /request─────────►│                        │
  │                         ├───add_message(cb)─────►│
  │                         │◄──callback(delta)──────┤
  │◄──SSE: delta────────────┤                        │
  │                         │    [tool call parsed]  │
  │◄──SSE: tool_call────────┤                        │
  │                         │                        │
  │                         │ ┌─execute_tool()       │
  │                         │ │  (local)             │
  │                         │ └────────────────────► │
  │◄──SSE: tool_result──────┤                        │
  │                         │                        │
  │                         ├───add_message(result)─►│
  │                         │◄──callback(delta)──────┤
  │◄──SSE: delta────────────┤                        │
  │◄──SSE: complete─────────┤                        │
```

## Differences from API Server

| Feature | CLI Server | API Server |
|---------|------------|------------|
| Tool Execution | Local | Client-side |
| Purpose | Remote prompt input, local tools | Full OpenAI compatibility |
| Use Case | Secure tool environment | API compatibility |
| SSE Events | Typed (delta, tool_call, etc) | OpenAI format chunks |

## CLI Client Backend

For connecting to remote CLI servers:

```cpp
class CLIClientBackend : public ApiBackend {
    Response add_message(Session& session, Message::Type type,
                        const std::string& content,
                        EventCallback callback = nullptr, ...) override;
private:
    std::string base_url;
    void sse_listener_thread();  // Receives server events
};
```

The SSE listener receives typed events and displays them:
- `delta` → `tio.write(content, Message::ASSISTANT)`
- `tool_call` → `tio.write("* name(...)", Message::TOOL_REQUEST)`
- `tool_result` → `tio.write("✓ ...", Message::TOOL_RESPONSE)`

**Note**: CLIClientBackend writes directly to tio because it's a proxy displaying remote server events, not generating content locally.

## Security Considerations

The CLI server executes tools locally with the permissions of the shepherd process. Bind to localhost (`127.0.0.1`) unless network access is intended and secured.

## Version History

- **2.13.0** - Unified EventCallback (no add_message_stream)
- **2.7.0** - Added async request queuing
- **2.6.1** - Added streaming support with SSE
- **2.6.0** - Initial CLI server implementation
