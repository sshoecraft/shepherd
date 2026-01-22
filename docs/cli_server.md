# CLI Server Module Documentation

## Overview

The CLI Server provides an HTTP server mode that executes tools locally while accepting prompts from remote clients. This enables a local machine with tools (filesystem access, shell commands) to serve as a backend for remote interfaces.

## Use Cases

### 🏠 Home Server AI Assistant

Run Shepherd 24/7 on a home server with full tool access:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Home Server                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Shepherd CLI Server                         │    │
│  │  • 2x RTX 3090 (48GB VRAM)                              │    │
│  │  • Qwen 72B model loaded                                │    │
│  │  • Full filesystem + shell access                       │    │
│  │  • Database credentials loaded                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │ SSE Stream
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────v────┐        ┌─────v────┐        ┌────v────┐
    │ Laptop  │        │  Phone   │        │ Tablet  │
    │ Client  │        │  Client  │        │ Client  │
    └─────────┘        └──────────┘        └─────────┘
```

All clients see the same conversation and can interact with it. Tools execute on the server where credentials and resources live.

### 🔒 Secure Tool Environment

- Database queries without exposing credentials to clients
- File operations on server filesystem
- Shell commands in controlled environment
- MCP servers running server-side

### 🔬 Research & DevOps

- Long-running research assistant that maintains context
- DevOps assistant with access to production systems
- Multi-user collaboration on same AI session

## Architecture

### Files

- `frontends/cli_server.h` - Class declaration
- `frontends/cli_server.cpp` - Implementation
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

## Configuration

### Server Configuration

Provider configuration for CLI server:

```json
{
  "name": "home-server",
  "backend": "llamacpp",
  "model": "/models/qwen3-72b-instruct.gguf",
  "context_size": 65536,
  "gpu_layers": 99
}
```

Start with:
```bash
./shepherd --cliserver --provider home-server --port 8000 --host 0.0.0.0
```

### Client Configuration

Add a CLI client provider:

```bash
shepherd provider add remote cli --api-base http://192.168.1.100:8000
```

Or in config:
```json
{
  "name": "remote",
  "backend": "cli",
  "api_base": "http://192.168.1.100:8000"
}
```

Connect:
```bash
./shepherd --provider remote
```

## Multi-Client Example

**Terminal 1 (Server):**
```bash
$ ./shepherd --cliserver --port 8000

CLI Server started on 0.0.0.0:8000
Waiting for connections...
```

**Terminal 2 (Client A):**
```bash
$ ./shepherd --provider remote

> What's the current working directory?
* shell(command="pwd")
/home/user/projects

> Create a file called test.txt
* write_file(path="test.txt", content="Hello from client A")
File created successfully.
```

**Terminal 3 (Client B - sees the conversation):**
```bash
$ ./shepherd --provider remote

# Shows existing conversation history via SSE
[Previous] What's the current working directory?
[Previous] * shell(command="pwd")
[Previous] /home/user/projects
[Previous] Create a file called test.txt
[Previous] File created successfully.

> Read test.txt
* read_file(path="test.txt")
Hello from client A
```

Both clients are interacting with the same session. Client B sees everything Client A did, and vice versa.

## Version History

- **2.13.0** - Unified EventCallback (no add_message_stream)
- **2.7.0** - Added async request queuing
- **2.6.1** - Added streaming support with SSE
- **2.6.0** - Initial CLI server implementation
