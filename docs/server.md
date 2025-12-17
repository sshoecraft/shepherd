# Server Module

## Overview

The Server module (`server/`) provides HTTP-based interfaces for Shepherd. There are two server types:

- **CLIServer**: Full-featured server with local tool execution and SSE streaming
- **APIServer**: OpenAI-compatible API endpoint

Both servers use the unified EventCallback pattern introduced in v2.13.0.

## Class Hierarchy

```
Frontend (abstract)
    │
    └── Server (abstract)
            │
            ├── CLIServer
            │   └── HTTP server + SSE + local tools
            │
            └── APIServer
                └── OpenAI-compatible /v1/chat/completions
```

## Server Base Class

The `Server` class (`server/server.h`, `server/server.cpp`) provides common functionality:

```cpp
class Server : public Frontend {
public:
    Server(const std::string& host, int port);
    virtual ~Server();

    // HTTP server management
    virtual bool start();
    virtual void stop();
    bool is_running() const;

protected:
    std::string host;
    int port;
    std::atomic<bool> running{false};

    // TCP server for main API
    httplib::Server tcp_server;

    // Unix socket for control commands
    httplib::Server control_server;

    // Subclasses implement
    virtual void register_endpoints(Session& session) = 0;
    virtual void add_status_info(nlohmann::json& status) {}
    virtual void on_server_start() {}
    virtual void on_server_stop() {}
};
```

### Control Socket

Local control commands via Unix domain socket:

- **Path**: `/var/tmp/shepherd.sock` (fallback: `/tmp/shepherd.sock`)
- **Endpoints**:
  - `GET /status` - Server status JSON
  - `POST /shutdown` - Graceful shutdown

### Common TCP Endpoints

- `GET /health` - Basic health check
- `GET /status` - Server status with model info

## CLIServer

Full-featured HTTP server that executes tools locally.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/request` | Submit user prompt |
| GET | `/updates` | SSE stream for real-time updates |
| GET | `/status` | Server status |

### EventCallback Pattern (v2.13.0)

CLIServer uses the unified EventCallback for streaming:

```cpp
void CLIServer::handle_request(const HttpRequest& req, HttpResponse& resp) {
    // Event callback streams to SSE clients
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

    // Single add_message() call with callback
    Response resp = session->add_message(
        Message::USER, prompt, stream_callback);
}
```

### SSE Events

Events broadcast to connected clients:

```json
// Text delta
{"type": "delta", "content": "Hello"}

// Tool call (server executes locally)
{"type": "tool_call", "name": "read_file", "args": {"path": "/foo"}}

// Tool result
{"type": "tool_result", "name": "read_file", "success": true, "content": "..."}

// Generation complete
{"type": "response_complete", "finish_reason": "stop"}

// Error
{"type": "error", "message": "Rate limited"}
```

### Tool Execution Flow

```
Client                   CLIServer                Backend
  │                         │                        │
  ├──POST /request─────────►│                        │
  │                         ├───add_message(cb)─────►│
  │                         │◄──callback(ASSISTANT)──┤
  │◄──SSE: delta────────────┤                        │
  │                         │    [tool call parsed]  │
  │◄──SSE: tool_call────────┤                        │
  │                         │                        │
  │                         │ ┌─execute_tool()       │
  │                         │ │  (local)             │
  │                         │ └─────────────────────►│
  │◄──SSE: tool_result──────┤                        │
  │                         │                        │
  │                         ├───add_message(result)─►│
  │                         │◄──callback(ASSISTANT)──┤
  │◄──SSE: delta────────────┤                        │
  │◄──SSE: complete─────────┤                        │
```

### Request Format

```json
{
    "prompt": "Read the file /etc/hostname",
    "stream": true
}
```

### Response (non-streaming)

```json
{
    "success": true,
    "response": "The contents of /etc/hostname are: myhost"
}
```

## APIServer

OpenAI-compatible API server.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

### EventCallback for Streaming

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

### Request Format

```json
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"}
    ],
    "stream": true,
    "max_tokens": 1000,
    "temperature": 0.7
}
```

### Response (non-streaming)

```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25
    }
}
```

### Streaming Response

```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":"!"}}]}

data: {"choices":[{"finish_reason":"stop"}]}

data: [DONE]
```

## CLIClient Backend

For remote CLI server connections, `CLIClientBackend` acts as a proxy:

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

### SSE Listener

CLIClientBackend spawns a thread to receive server events:

```cpp
void CLIClientBackend::sse_listener_thread() {
    while (sse_running) {
        std::string event = read_sse_event();
        auto json = nlohmann::json::parse(event);

        // Display based on event type
        if (json["type"] == "delta") {
            tio.write(json["content"], Message::ASSISTANT);
        } else if (json["type"] == "tool_call") {
            tio.write("* " + json["name"].get<std::string>() + "(...)\n",
                     Message::TOOL_REQUEST);
        } else if (json["type"] == "tool_result") {
            tio.write("✓ Success\n", Message::TOOL_RESPONSE);
        }
    }
}
```

**Note**: CLIClientBackend is the only backend that writes directly to tio - because it's a proxy displaying events from a remote server, not generating content locally.

## Thread Safety

Both servers handle concurrent requests with session locking:

```cpp
struct SessionState {
    std::unique_ptr<Session> session;
    std::mutex session_mutex;
};

void handle_request(...) {
    std::lock_guard<std::mutex> lock(state.session_mutex);
    // Process request safely
}
```

## Control Client

The control client communicates with running servers via Unix socket:

```bash
# Get server status
shepherd ctl status

# Shutdown server gracefully
shepherd ctl shutdown

# Specify socket explicitly
shepherd ctl status --socket /tmp/shepherd-api.sock
```

## Command Line

```bash
# Start API server
shepherd --apiserver --host 0.0.0.0 --port 8000

# Start CLI server
shepherd --cliserver --host 0.0.0.0 --port 8000
```

## Error Handling

```json
{
    "error": {
        "message": "Error description",
        "type": "invalid_request_error",
        "code": "400"
    }
}
```

| Status | Type | When |
|--------|------|------|
| 400 | `invalid_request_error` | Invalid JSON, context limit |
| 500 | `server_error` | Backend errors, exceptions |

## Files

- `server/server.h` / `server/server.cpp` - Base server class
- `server/cli_server.h` / `server/cli_server.cpp` - CLI server with tools
- `server/api_server.h` / `server/api_server.cpp` - OpenAI-compatible API
- `server/control.h` / `server/control.cpp` - Control client
- `backends/cli_client.h` / `backends/cli_client.cpp` - Remote CLI client

## Version History

- **2.13.0** - Unified EventCallback pattern (no more add_message_stream)
- **2.10.0** - Added SSE support for real-time updates
- **2.6.0** - Initial server implementations
