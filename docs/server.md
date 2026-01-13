# Server Module

## Overview

The Server module provides HTTP-based interfaces for Shepherd. There are two server types:

- **CLIServer**: Full-featured server with local tool execution and SSE streaming
- **APIServer**: OpenAI-compatible API endpoint

Both servers inherit from the `Server` base class and use the unified EventCallback pattern.

## Class Hierarchy

```
Frontend (abstract)
    |
    +-- Server (abstract)
            |
            +-- CLIServer
            |   +-- HTTP server + SSE + local tool execution
            |   +-- See docs/cliserver.md for details
            |
            +-- APIServer
                +-- OpenAI-compatible /v1/chat/completions
```

## Server Base Class

The `Server` class (`server.h`, `server.cpp`) provides common functionality:

```cpp
class Server : public Frontend {
public:
    Server(const std::string& host, int port, const std::string& server_type);
    virtual ~Server();

    // Main run loop - starts TCP and control socket
    int run(Provider* cmdline_provider = nullptr) override;

    // Graceful shutdown
    void shutdown();

protected:
    // Subclasses implement
    virtual void register_endpoints() = 0;
    virtual void add_status_info(nlohmann::json& status) {}
    virtual void on_server_start() {}
    virtual void on_server_stop() {}
    virtual void on_shutdown() {}

    // TCP server for main API
    httplib::Server tcp_server;

    // Unix socket for control commands
    httplib::Server control_server;

    // Server state
    std::atomic<bool> running{true};
    std::chrono::steady_clock::time_point start_time;
    std::atomic<uint64_t> requests_processed{0};

    // Configuration
    std::string host;
    int port;
    std::string server_type;
    std::string control_socket_path;
};
```

### Server Lifecycle

```cpp
int Server::run(Provider* cmdline_provider) {
    // 1. Connect to provider
    connect_provider(provider_name);

    // 2. Configure session
    session.desired_completion_tokens = calculate_desired_completion_tokens(...);
    session.auto_evict = false;  // Server mode returns errors instead

    // 3. Register common endpoints (/health, /status)
    register_common_endpoints();

    // 4. Let subclass register its endpoints
    register_endpoints();

    // 5. Start control socket
    start_control_socket();

    // 6. Let subclass start background threads
    on_server_start();

    // 7. Start TCP server (blocks until stopped)
    tcp_server.listen(host, port);

    // 8. Cleanup
    on_server_stop();
    cleanup_control_socket();
}
```

### Control Socket

Local control commands via Unix domain socket:

- **Path**: `/var/tmp/shepherd-<port>.sock` (fallback: `/tmp/shepherd-<port>.sock`)
- Each server instance gets its own socket based on its listening port
- **Endpoints**:
  - `GET /status` - Server status JSON
  - `POST /shutdown` - Graceful shutdown

### Common TCP Endpoints

Both CLIServer and APIServer inherit these:

- `GET /health` - Basic health check
- `GET /status` - Server status with model info

## CLIServer

Full-featured HTTP server that executes tools locally.

**See [docs/cliserver.md](cliserver.md) for complete documentation including:**
- HTTP endpoints (/request, /updates, /session, /clear)
- SSE broadcasting architecture
- Tool execution flow
- Multi-client support

### Quick Reference

```bash
# Start CLI server
shepherd --cliserver --host 0.0.0.0 --port 8000

# Connect as client
shepherd --provider cli
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
auto stream_callback = [&](CallbackEvent event,
                           const std::string& content,
                           const std::string& name,
                           const std::string& id) -> bool {
    if (event == CallbackEvent::CONTENT) {
        // Format as OpenAI chunk
        nlohmann::json chunk = {
            {"choices", {{
                {"delta", {{"content", content}}}
            }}}
        };
        send_sse("data: " + chunk.dump() + "\n\n");
    }
    return true;
};

backend->callback = stream_callback;
session.add_message(Message::USER, prompt);
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

## Control Client

The control client communicates with running servers via Unix socket:

```bash
# Get server status (default port 8000)
shepherd ctl status

# Get status for server on specific port
shepherd ctl status 8080

# Shutdown server gracefully (default port 8000)
shepherd ctl shutdown

# Shutdown server on specific port
shepherd ctl shutdown 8080

# Specify socket explicitly (overrides port)
shepherd ctl status --socket /tmp/shepherd-8080.sock
```

## Command Line

```bash
# Start API server
shepherd --apiserver --host 0.0.0.0 --port 8000

# Start CLI server
shepherd --cliserver --host 0.0.0.0 --port 8000

# Control running server
shepherd ctl status
shepherd ctl shutdown
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

- `server.h` / `server.cpp` - Base Server class
- `frontends/api_server.h` / `frontends/api_server.cpp` - API server
- `frontends/cli_server.h` / `frontends/cli_server.cpp` - CLI server

## Version History

- **2.18.0** - Port-based control sockets (shepherd-<port>.sock) for multiple server instances
- **2.13.0** - Unified EventCallback pattern
- **2.10.0** - Added SSE support for real-time updates
- **2.6.0** - Initial server implementations
