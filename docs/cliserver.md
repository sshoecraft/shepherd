# CLI Server and CLI Client Architecture

## Overview

The CLI Server/Client architecture enables remote interaction with a shepherd instance that executes tools locally. This is useful when:
- Tools require local execution (filesystem access, shell commands)
- Multiple clients need to share a single LLM session
- Remote interfaces need to drive a local shepherd instance

## Design Principle: Session Mirroring

**CLI clients connected to a CLI server mirror the server's session.** Everything that happens on the server, all connected clients see:
- All deltas from model generation
- All tool calls and results
- All message additions
- All completions and errors

The `/updates` SSE endpoint is the **primary display channel** for CLI clients. The POST `/request` endpoint is for submitting prompts - streaming or batched response depending on client preference.

## Architecture Diagram

```
                              CLI SERVER SIDE
  (running: shepherd --cliserver --port 8000)

  +----------------+     +--------------+     +-------------------------+
  |   CLIServer    |---->|   Backend    |---->|  LLM (LlamaCpp,        |
  |   (Frontend)   |     |  (any type)  |     |  OpenAI, Anthropic)    |
  +-------+--------+     +--------------+     +-------------------------+
          |
          | owns
          v
  +----------------+
  |    Session     |  <-- Single session, shared by all clients
  |  (messages,    |
  |   tokens)      |
  +----------------+
          |
          | uses
          v
  +----------------+
  |     Tools      |  <-- Executes locally on server
  |  (filesystem,  |
  |   shell, etc)  |
  +----------------+

  HTTP Endpoints:
    POST /request  --> Streaming (SSE) or Batched (JSON) response
    GET  /updates  --> SSE broadcast to all observers
    GET  /session  --> Session state JSON
    POST /clear    --> Clear context

============================================================================

                              CLI CLIENT SIDE
  (running: shepherd --provider cli)

  +----------------+     +----------------------+
  |      CLI       |---->|  CLIClientBackend    |
  |   (Frontend)   |     |  (acts as Backend)   |
  +----------------+     +----------+-----------+
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
             +------------------+    +------------------+
             |  POST /request   |    |  GET /updates    |
             |  (stream=false)  |    |  (SSE observer)  |
             +------------------+    +------------------+
                    |                       |
                    v                       v
             Blocks, returns         Receives all events,
             complete JSON           handles display
```

## ClientOutput Abstraction

The server uses a unified `ClientOutput` interface to handle different output modes:

```cpp
class ClientOutput {
public:
    virtual void on_delta(const std::string& delta) = 0;
    virtual void on_user_prompt(const std::string& prompt) = 0;
    virtual void on_message_added(const std::string& role, const std::string& content, int tokens) = 0;
    virtual void on_tool_call(const std::string& name, const json& params, const std::string& id) = 0;
    virtual void on_tool_result(const std::string& name, bool success, const std::string& error) = 0;
    virtual void on_complete(const std::string& full_response) = 0;
    virtual void on_error(const std::string& error) = 0;
    virtual void flush() = 0;
    virtual bool is_connected() const = 0;
};
```

### Implementations

**StreamingOutput**: Writes SSE events immediately to an httplib::DataSink
- Used for POST /request with stream=true
- Used for /updates observers

**BatchedOutput**: Accumulates content, writes complete JSON on flush()
- Used for POST /request with stream=false

### Data Flow

```
                    Backend generates tokens
                            |
                            v
                    Backend callback fires
                            |
                            v
                do_generation() iterates all outputs:
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
    StreamingOutput   StreamingOutput   BatchedOutput
    (/updates obs)    (/updates obs)    (POST requester)
            |               |               |
            v               v               v
    SSE: all events   SSE: all events   accumulate
    immediately       immediately       until done
```

## CLI Server (CLIServer)

### Files
- `frontends/cli_server.h` - Class declaration
- `frontends/cli_server.cpp` - Implementation
- `frontends/client_output.h` - ClientOutput interface
- `frontends/client_output.cpp` - StreamingOutput, BatchedOutput implementations

### CliServerState Structure

```cpp
struct CliServerState {
    CLIServer* server;      // For calling execute_tool
    Backend* backend;       // The LLM backend
    Session* session;       // Conversation state
    Tools* tools;           // Tool registry

    // Input queue for async requests
    std::deque<std::string> input_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    // Request mutex - ensures only one request at a time
    std::mutex request_mutex;

    // SSE observer clients (connected to /updates)
    std::vector<ClientOutputs::StreamingOutput*> observers;
    std::mutex observers_mutex;

    std::atomic<bool> processing{false};
    std::atomic<bool> running{true};
    std::string current_request;

    // Methods
    void add_input(const std::string& prompt);
    size_t queue_size();
    std::string get_next_input();
    void send_to_observers(const std::function<void(ClientOutput&)>& action);
    void register_observer(StreamingOutput* observer);
    void unregister_observer(StreamingOutput* observer);
};
```

### HTTP Endpoints

#### POST /request

Main request endpoint. Supports three modes:

| Mode | Parameters | Behavior |
|------|------------|----------|
| Async | `async=true, stream=false` | Queue request, return immediately |
| Streaming | `stream=true` | SSE response + broadcast to observers |
| Batched | `stream=false` | Wait for complete JSON response + broadcast to observers |

**Request Format:**
```json
{
    "prompt": "What files are in the current directory?",
    "stream": false,
    "async": false,
    "max_tokens": 4096
}
```

**Parameters:**
- `prompt` (required): The user message to process
- `stream` (optional, default: false): If true, return SSE stream; if false, return complete JSON
- `async` (optional, default: false): If true, queue request and return immediately
- `max_tokens` (optional, default: 0): Maximum generation tokens. 0 = auto, -1 = maximum available, >0 = explicit limit. Overrides server's default.

**Streaming Response (SSE when stream=true):**
```
data: {"type": "delta", "data": {"delta": "Let me"}}

data: {"type": "tool_call", "data": {"tool_call": "list_files", "parameters": {"path": "."}}}

data: {"type": "tool_result", "data": {"tool_name": "list_files", "success": true}}

data: {"type": "response_complete", "data": {"response": "..."}}

data: {"done": true}
```

**Batched Response (JSON when stream=false):**
```json
{
    "success": true,
    "response": "The directory contains: file1.txt, file2.txt"
}
```

#### GET /updates

SSE endpoint for session observers. Clients connect and receive ALL events during generation.

**Event Types:**
```json
{"type": "delta", "data": {"delta": "Hello!"}}
{"type": "message_added", "data": {"role": "user", "content": "hello"}}
{"type": "tool_call", "data": {"tool_call": "read_file", "parameters": {...}}}
{"type": "tool_result", "data": {"tool_name": "read_file", "success": true}}
{"type": "response_complete", "data": {"response": "..."}}
{"type": "error", "data": {"error": "..."}}
{"type": "connected", "data": {"client_id": "127.0.0.1"}}
```

#### GET /session

Returns full session state including all messages.

#### POST /clear

Clears conversation history.

### Unified Generation Flow

All requests (streaming and batched) go through `do_generation()`:

```cpp
void do_generation(CliServerState& state, ClientOutput* requester, const std::string& prompt) {
    // Send to requester AND all observers
    requester->on_user_prompt(prompt);
    state.send_to_observers([&](ClientOutput& obs) { obs.on_user_prompt(prompt); });

    // Backend callback sends to all
    auto callback = [&](CallbackEvent type, const std::string& content, ...) -> bool {
        if (type == CallbackEvent::CONTENT) {
            requester->on_delta(content);
            state.send_to_observers([&](ClientOutput& obs) { obs.on_delta(content); });
        }
        // ... handle other events ...
        return true;
    };

    state.backend->callback = callback;
    state.session->add_message(Message::USER, prompt);

    // Tool execution loop...

    requester->on_complete(accumulated_response);
    state.send_to_observers([&](ClientOutput& obs) { obs.on_complete(accumulated_response); });
    requester->flush();
}
```

## CLI Client Backend (CLIClientBackend)

### Files
- `backends/cli_client.h` - Class declaration
- `backends/cli_client.cpp` - Implementation

### Key Behavior

1. **Constructor**: Fetches session from server, starts SSE listener on `/updates`
2. **SSE Listener**: Handles ALL display - deltas, tool calls, results, completions
3. **send_request()**: Uses `stream=false`, blocks until complete, returns JSON
4. **No duplicate output**: SSE listener is the sole display path

### send_request Method

```cpp
Response send_request(const std::string& prompt, int max_tokens, EventCallback callback) {
    json request;
    request["prompt"] = prompt;
    if (max_tokens != 0) {
        request["max_tokens"] = max_tokens;
    }
    // stream not set - let server use its default

    // Simple synchronous POST
    HttpResponse http_resp = http_client->post(endpoint, request.dump(), headers);

    // Parse JSON response
    json response = json::parse(http_resp.body);
    resp.success = response.value("success", false);
    resp.content = response.value("response", "");

    return resp;
}
```

### SSE Listener Thread

Handles all output display:

```cpp
void sse_listener_thread() {
    while (sse_running) {
        // Connect to /updates
        // For each SSE event:
        if (event_type == "delta") {
            std::cout << delta;
        } else if (event_type == "tool_call") {
            std::cout << "  * " << tool_name << "(...)\n";
        } else if (event_type == "tool_result") {
            std::cout << (success ? "    Success\n" : "    Error: ...\n");
        } else if (event_type == "response_complete") {
            std::cout << "\n";
        }
    }
}
```

## Usage

### Starting CLI Server

```bash
# Start with default settings
shepherd --cliserver --port 8000

# Specify host and provider
shepherd --cliserver --host 0.0.0.0 --port 8000 --provider my-provider
```

### Connecting as CLI Client

Configure provider in `~/.config/shepherd/providers.json`:
```json
{
    "cli": {
        "type": "cli",
        "api_base": "http://localhost:8000"
    }
}
```

Then connect:
```bash
shepherd --provider cli
```

### Multi-Client Scenario

```
Terminal 1: shepherd --cliserver --port 8000
Terminal 2: shepherd --provider cli  (connects, sends prompts)
Terminal 3: shepherd --provider cli  (connects, observes via /updates)
```

All clients see all activity in real-time via the `/updates` SSE stream.

### External API Client

```bash
# Streaming request
curl -X POST http://localhost:8000/request \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "stream": true}'

# Batched request
curl -X POST http://localhost:8000/request \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "stream": false}'
```

## Thread Safety

- `request_mutex` ensures only one request processes at a time
- `observers_mutex` protects the observer list during iteration
- `queue_mutex` + `queue_cv` for async request queue

## Version History

- **2.17.0** - Added max_tokens parameter to /request endpoint, flows through to backend
- **2.16.0** - Unified ClientOutput architecture, fixed duplicate output
- **2.15.0** - Previous architecture with dual streaming paths
- **2.13.0** - Unified EventCallback pattern
- **2.10.0** - Added SSE /updates endpoint
- **2.6.0** - Initial CLI server implementation
