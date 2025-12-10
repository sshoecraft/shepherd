# Shepherd Server Architecture

## Overview

The Shepherd server system provides HTTP APIs for inference using local backends (llama.cpp, TensorRT-LLM). There are two server types:

- **API Server** - OpenAI-compatible HTTP API for external clients
- **CLI Server** - HTTP API with local tool execution for remote agent control

Both servers share a common base class (`Server`) that handles HTTP infrastructure, control socket, and lifecycle management.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Server (Base Class)                      │
│  - TCP server for main API endpoints                             │
│  - Unix socket for control commands (shutdown, status)           │
│  - Common endpoints: /health, /status                            │
│  - Lifecycle management (start, stop, cleanup)                   │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
        ┌───────────┴──────────┐    ┌───────────┴──────────┐
        ▼                      ▼    ▼                      ▼
┌───────────────────┐  ┌───────────────────┐
│    APIServer      │  │    CLIServer      │
│                   │  │                   │
│  OpenAI-compat:   │  │  Tool execution:  │
│  /v1/chat/compl.  │  │  /request         │
│  /v1/models       │  │  /clear           │
│                   │  │                   │
│  No tool exec     │  │  Async queue      │
│  (client does it) │  │  Processor thread │
└───────────────────┘  └───────────────────┘
```

## Server Base Class

The `Server` class (`server/server.h`, `server/server.cpp`) provides common functionality:

### Members

- `tcp_server` - httplib::Server for main API endpoints
- `control_server` - httplib::Server for Unix socket control
- `running` - Atomic bool for shutdown coordination
- `start_time` - Server start time for uptime tracking
- `requests_processed` - Request counter

### Virtual Methods

Subclasses must implement:
- `register_endpoints(Session& session)` - Register server-specific endpoints

Subclasses may override:
- `add_status_info(json& status)` - Add fields to status response
- `on_server_start()` - Called before listen starts (e.g., start processor thread)
- `on_server_stop()` - Called after listen stops (e.g., join threads)

### Control Socket

The server creates a Unix domain socket for local control commands:

- **Path**: `/var/tmp/shepherd.sock` (falls back to `/tmp/shepherd.sock` if `/var/tmp` not writable)
- **Permissions**: 0600 (user-only)
- **Endpoints**:
  - `GET /status` - Server status JSON
  - `POST /shutdown` - Initiate graceful shutdown

### Common TCP Endpoints

- `GET /health` - Basic health check
- `GET /status` - Server status with model info

## API Server

The API server (`server/api_server.h`, `server/api_server.cpp`) provides an OpenAI-compatible HTTP API.

### Endpoints

#### POST /v1/chat/completions

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
  "top_p": 0.9
}
```

**Response:**
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

#### GET /v1/models

List available models.

#### GET /v1/models/:model_name

Get specific model info.

## CLI Server

The CLI server (`server/cli_server.h`, `server/cli_server.cpp`) provides tool execution for remote agent control.

### Endpoints

#### POST /request

Submit a prompt for processing with tool execution.

**Request:**
```json
{
  "prompt": "Your prompt here",
  "stream": false,
  "async": false
}
```

- `stream: true` - Use Server-Sent Events for streaming response
- `async: true` - Queue request and return immediately

**Response (sync):**
```json
{
  "success": true,
  "response": "Final response after tool execution"
}
```

**Response (async):**
```json
{
  "success": true,
  "queued": true,
  "queue_position": 1
}
```

#### POST /clear

Reset conversation history.

### Processor Thread

The CLI server runs a background thread that processes queued async requests:

```cpp
void CLIServer::on_server_start() {
    processor_thread = std::thread([this]() {
        while (state.running) {
            std::string prompt = state.get_next_input();
            // Process request with tool execution loop
        }
    });
}
```

## Control Client (shepherd ctl)

The control client (`server/control.h`, `server/control.cpp`) communicates with running servers via Unix socket.

### Usage

```bash
# Get server status
shepherd ctl status

# Shutdown server gracefully
shepherd ctl shutdown

# Specify socket explicitly
shepherd ctl status --socket /tmp/shepherd-api.sock
```

### Auto-detection

If no socket is specified, the client tries:
1. `/var/tmp/shepherd.sock`
2. `/tmp/shepherd.sock`

## Streaming

Both servers support Server-Sent Events for streaming:

```
data: {"delta": "Hello"}

data: {"delta": " world"}

data: {"done": true, "response": "Hello world"}
```

## Concurrency

The API server uses a mutex to serialize backend requests:

```cpp
std::mutex backend_mutex;

tcp_server.Post("/v1/chat/completions", [this](...) {
    std::lock_guard<std::mutex> lock(backend_mutex);
    // ... handle request
});
```

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
| 500 | `server_error` | Backend errors, exceptions |

## Configuration

### Command Line

```bash
# Start API server
shepherd --apiserver --host 0.0.0.0 --port 8000

# Start CLI server
shepherd --cliserver --host 0.0.0.0 --port 8000
```

### Control Socket Path

Control socket path: `/var/tmp/shepherd.sock` (or `/tmp/shepherd.sock` as fallback)

## Key Files

- `server/server.h` - Server base class interface
- `server/server.cpp` - Server base class implementation
- `server/api_server.h` - API server interface
- `server/api_server.cpp` - API server implementation
- `server/cli_server.h` - CLI server interface
- `server/cli_server.cpp` - CLI server implementation
- `server/control.h` - Control client interface
- `server/control.cpp` - Control client implementation
