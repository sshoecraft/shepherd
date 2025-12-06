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

    int run(std::unique_ptr<Backend>& backend, Session& session) override;
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

**Response (streaming):**

Server-Sent Events (SSE) format:
```
data: {"delta": "Hello"}

data: {"delta": " world"}

data: {"done": true, "response": "Hello world"}
```

**Response (non-streaming):**
```json
{
    "response": "Hello world",
    "success": true
}
```

## Streaming Support

The CLI server supports real-time token streaming using SSE. When `stream: true` is set in the request:

1. Server calls `session->add_message_stream()` with a streaming callback
2. Each generated token is sent as an SSE event with the delta
3. Final event includes `done: true` and the complete response
4. Connection closes immediately after the done event

### Backend Requirements

For streaming to work, the backend must implement `add_message_stream()`:

| Backend | Streaming Support |
|---------|------------------|
| LlamaCpp | Yes (passes callback to generate) |
| TensorRT | Yes (passes callback to generate) |
| Anthropic | Yes (SSE parsing) |
| OpenAI | Yes (SSE parsing) |
| Gemini | Yes (SSE parsing via streamGenerateContent) |
| Ollama | Yes (NDJSON parsing) |

Backends without `add_message_stream()` fall back to the base `Backend::add_message_stream()` which calls `add_message()` and returns the complete response in a single callback invocation.

## Differences from API Server

| Feature | CLI Server | API Server |
|---------|------------|------------|
| Tool Execution | Local | Client-side |
| Purpose | Remote prompt input, local tools | Full OpenAI compatibility |
| Use Case | Secure tool environment | API compatibility |
| Streaming Format | SSE with delta/done | OpenAI SSE chunks |

## Security Considerations

The CLI server executes tools locally with the permissions of the shepherd process. Bind to localhost (`127.0.0.1`) unless network access is intended and secured.

## Implementation Notes

### Connection Closing

The content provider lambda must return `false` after calling `sink.done()` to signal httplib that the response is complete. Returning `true` causes httplib to wait for more content, resulting in a delay before the connection closes.

```cpp
sink.done();
return false;  // Signal no more content
```

### Client Stream Handling

The CLI client (`cli_client.cpp`) uses `post_stream_cancellable()` for streaming requests. The stream handler always returns `true` to curl to avoid "Failed writing" errors, using a `stream_complete` flag internally to track completion.

## Version History

- **2.6.0** - Initial CLI server implementation
- **2.6.1** - Added streaming support with SSE, CLI client backend, and add_message_stream for all backends
