# Shepherd Architecture

## Overview

Shepherd is an LLM chat application with support for multiple backends (local and API), tool execution, and multiple frontend modes (CLI, TUI, HTTP servers). Version 2.15.0 introduced a pure callback architecture for backend-to-frontend communication.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ┌─────────┐  ┌─────────────┐  ┌───────────┐  ┌───────────────┐│
│  │   CLI   │  │  CLIServer  │  │ APIServer │  │  TUI (visual) ││
│  └────┬────┘  └──────┬──────┘  └─────┬─────┘  └───────────────┘│
│       │              │               │                          │
│       └──────────────┼───────────────┘                          │
│                      │                                          │
│              ┌───────┴───────┐                                  │
│              │   Frontend    │  (base class)                    │
│              │  init_tools() │                                  │
│              └───────┬───────┘                                  │
└──────────────────────┼──────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────┐
│                      │         Session Layer                    │
│              ┌───────┴───────┐                                  │
│              │    Session    │                                  │
│              │ add_message() │ ← EventCallback (optional)       │
│              └───────┬───────┘                                  │
└──────────────────────┼──────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────┐
│                      │         Backend Layer                    │
│              ┌───────┴───────┐                                  │
│              │    Backend    │  (abstract base)                 │
│              │ add_message() │ → EventCallback invocations      │
│              └───────┬───────┘                                  │
│       ┌──────────────┼──────────────────┐                       │
│  ┌────┴────┐  ┌──────┴──────┐  ┌────────┴────────┐             │
│  │ OpenAI  │  │  LlamaCpp   │  │    TensorRT     │  ...        │
│  └─────────┘  └─────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────┐
│                      │      Presentation Layer                  │
│              ┌───────┴───────┐                                  │
│              │  TerminalIO   │  (input + output)                │
│              │   TUI class   │  (ncurses visual)                │
│              └───────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Pure Callback Architecture (v2.15.0)

### CallbackEvent Enum

All backend-to-frontend communication flows through callbacks using a dedicated event enum:

```cpp
// Defined in backend.h
enum class CallbackEvent {
    CONTENT,      // Assistant text chunk
    THINKING,     // Reasoning/thinking chunk (if show_thinking enabled)
    TOOL_CALL,    // Model requesting a tool call
    TOOL_RESULT,  // Result of tool execution (summary in content)
    USER_PROMPT,  // Echo user's prompt back
    SYSTEM,       // System info/status messages
    ERROR,        // Error occurred (message in content, type in name)
    STOP,         // Generation complete (finish_reason in content)
    CODEBLOCK,    // Content inside ``` code blocks (for special formatting)
    STATS         // Performance stats (prefill/decode speed, KV cache info)
};

// Defined in backends/backend.h
using EventCallback = std::function<bool(
    CallbackEvent event,          // Event type
    const std::string& content,   // Event content (delta text, error msg, finish_reason)
    const std::string& name,      // Tool name (TOOL_CALL), error type (ERROR)
    const std::string& id         // Tool call ID (for correlation)
)>;
// Returns: true = continue, false = cancel generation
```

### Message::Role Enum

Separate enum for messages stored in session (sent to APIs):

```cpp
// Defined in message.h
enum Role {
    SYSTEM,         // System prompt
    USER,           // User messages
    ASSISTANT,      // Provider responses
    TOOL_RESPONSE,  // Result of tool execution (modern "tool" role)
    FUNCTION        // Legacy OpenAI function result
};
```

### Message Flow

```
1. User Input
   └─→ Frontend receives input
       └─→ Backend callback set at construction
           └─→ Calls session.add_message(role, content, ...)
               └─→ Backend generates tokens
                   └─→ For each token: callback(CONTENT, delta, "", "")
                   └─→ For tool call: callback(TOOL_CALL, json, name, id)
                   └─→ On error: callback(ERROR, message, type, "")
                   └─→ On complete: callback(STOP, finish_reason, "", "")
               └─→ Backend updates session with messages
           └─→ add_message returns void
       └─→ Frontend checks pending tool calls from callbacks
           └─→ If tool call: execute, submit result, repeat
```

### Callback Invocation Points

Backends invoke the callback at these points:
- **Text delta**: `callback(CallbackEvent::CONTENT, delta_text, "", "")`
- **Code block**: `callback(CallbackEvent::CODEBLOCK, code_content, "", "")` (inside ```)
- **Thinking**: `callback(CallbackEvent::THINKING, reasoning, "", "")`
- **Tool call**: `callback(CallbackEvent::TOOL_CALL, tool_json, tool_name, tool_id)`
- **Error**: `callback(CallbackEvent::ERROR, error_message, error_type, "")`
- **Done**: `callback(CallbackEvent::STOP, finish_reason, "", "")`

### Tool Call Validation

The `output()` filter validates tool names against `Backend::valid_tool_names`:
- Frontends populate this set from `session.tools` at startup
- If a parsed tool name isn't in the set, content is emitted as CONTENT instead of TOOL_CALL
- This prevents model output that looks like tool calls from being filtered when it's not a valid tool

### Streaming vs Non-Streaming

The callback is always called - streaming controls granularity:
- **Streaming ON**: Callback receives deltas as they arrive
- **Streaming OFF**: Callback receives full content once, then STOP

```cpp
// Backend callback set at construction
Backend::EventCallback cb = [](CallbackEvent event, const std::string& content,
                               const std::string& name, const std::string& id) {
    switch (event) {
        case CallbackEvent::CONTENT:
            tio.write(content);  // Live output
            break;
        case CallbackEvent::TOOL_CALL:
            queue_tool_call(content, name, id);
            break;
        case CallbackEvent::STOP:
            // finish_reason in content: "stop", "tool_calls", "length", "error"
            break;
    }
    return true;
};

// add_message returns void - all output via callback
session.add_message(Message::USER, prompt);
```

## Key Components

### Frontend (frontend.h/cpp)
Base class for all presentation layers. Manages:
- **Session ownership** - Frontend owns the Session as a member variable
- Provider list and connection
- Backend lifecycle
- Common tool initialization via `init_tools()`

Subclasses:
- `CLI`: Interactive terminal mode
- `TUI`: NCurses visual mode
- `Server`: Base for HTTP servers
- `CLIServer`: HTTP server with tool execution
- `APIServer`: OpenAI-compatible API server

Key interface:
```cpp
static std::unique_ptr<Frontend> create(const std::string& mode, ...);
int run();  // Main loop - uses frontend's session member
bool connect_next_provider();  // Uses frontend's session member
```

### Session (session.h/cpp)
Manages conversation state. **Session is the source of truth for token counts.**
- Message history
- Token counting (`total_tokens`, `last_prompt_tokens`, `last_assistant_message_tokens`)
- Context eviction
- Tool definitions

Key method:
```cpp
void add_message(Message::Role role,
                const std::string& content,
                const std::string& tool_name = "",
                const std::string& tool_id = "",
                int max_tokens = 0);
// All output flows through backend callback (CONTENT, TOOL_CALL, ERROR, STOP)
// Backend updates session token counts during generation
```

Token fields in Session (authoritative source):
- `total_tokens` - Total tokens from API response
- `last_prompt_tokens` - Prompt tokens from last API call
- `last_assistant_message_tokens` - Completion tokens from last response

### Backend (backends/backend.h)
Abstract base for all LLM backends. Key responsibilities:
- Generate responses via `generate_from_session(Session& session, int max_tokens)`
- Update session token counts after generation
- Invoke callbacks for streaming output

Implementations:
- `OpenAIBackend`: OpenAI API
- `AnthropicBackend`: Claude API
- `GeminiBackend`: Google Gemini API
- `OllamaBackend`: Local Ollama
- `LlamaCppBackend`: Local llama.cpp
- `TensorRTBackend`: NVIDIA TensorRT-LLM
- `CLIClientBackend`: Proxy to remote CLI server

Token flow in generate_from_session():
```cpp
void generate_from_session(Session& session, int max_tokens) {
    // ... make API call / generate tokens ...

    // Update session token counts (session is source of truth)
    session.total_tokens = resp.prompt_tokens + resp.completion_tokens;
    session.last_prompt_tokens = resp.prompt_tokens;
    session.last_assistant_message_tokens = resp.completion_tokens;

    // Invoke callbacks for content
    callback(CallbackEvent::CONTENT, content, "", "");
    callback(CallbackEvent::STOP, finish_reason, "", "");
}
```

### TerminalIO (terminal_io.h/cpp)
Unified input/output handling:
- Raw terminal mode (termios)
- Input processing (escape sequences, history, paste)
- Output filtering (tool calls, thinking blocks)
- Writer functions (console, TUI, server)

### TUI (tui.h/cpp)
NCurses-based visual interface:
- Output window with scrollback
- Input box with cursor
- Status line
- Queued input display

Visual-only: TerminalIO handles actual input processing.

### GenerationThread (generation_thread.h/cpp)
Runs LLM generation in background thread:
- Prevents UI blocking
- Handles async generation requests
- Reports completion via polling

## Tool Execution

Tools are executed by the frontend, not the backend:

```
Backend generates response
    │
    └─→ Response contains tool_calls or parsed tool call markers
        │
        └─→ Frontend extracts tool call
            │
            └─→ Frontend executes tool via Tools class
                │
                └─→ Frontend submits tool result back to session
                    │
                    └─→ Repeat until no more tool calls
```

### Tool Call Filtering

During streaming, tool call syntax (JSON/XML) is filtered from display:
- Filtering happens in `TerminalIO::write()`
- Uses state machine to detect markers
- Accumulates tool call content silently
- Shows formatted version after completion

## Input Handling (v2.13.0)

Unified input in TerminalIO for both console and TUI modes:

```
TerminalIO::input_loop() [separate thread]
    │
    └─→ poll() + read() from stdin
        │
        └─→ process_input() - handles:
            ├─ Escape sequences (arrows, home/end, delete)
            ├─ Bracketed paste (multiline support)
            ├─ History navigation (up/down arrows)
            ├─ Line editing (Ctrl+U/K/A/E)
            ├─ Single escape (cancel generation)
            └─ Double escape (clear input)
        │
        └─→ submit_input() - adds to input queue
            │
            └─→ Main loop consumes via tio.read()
```

## Provider System

Providers are configured backends with connection info:

```cpp
struct Provider {
    std::string name;
    std::string backend;     // "openai", "anthropic", "ollama", etc.
    std::string model;
    std::string api_key;
    std::string api_base;
    int priority;
};
```

Provider connection:
1. `Frontend::create()` loads providers from disk
2. `connect_next_provider()` tries in priority order
3. On success, `frontend->backend` is set
4. `session.backend` points to same instance

## Server Modes

### CLI Server (frontends/cli_server.cpp)
- Endpoints: `/request`, `/updates` (SSE), `/session`, `/clear`
- Executes tools locally
- Streams via Server-Sent Events
- Maintains session state
- See [docs/cliserver.md](cliserver.md) for complete architecture

### API Server (frontends/api_server.cpp)
- OpenAI-compatible endpoints
- `/v1/chat/completions`
- `/v1/models`
- Stateless (prefix caching for efficiency)
- No tool execution (returns tool_calls to client)

### CLI Client Backend (backends/cli_client.cpp)
- Proxy backend for remote CLI server connections
- SSE listener thread for real-time updates from server
- Tool execution happens on server, not client
- See [docs/cliserver.md](cliserver.md) for architecture details

## File Structure

```
shepherd/
├── main.cpp              # Entry point, arg parsing
├── frontend.h/cpp        # Frontend base class
├── backend.h/cpp         # Backend base class + EventCallback
├── session.h/cpp         # Session management
├── server.h/cpp          # Server base class
├── terminal_io.h/cpp     # Input/output handling
├── frontends/
│   ├── cli.h/cpp         # CLI frontend
│   ├── tui.h/cpp         # NCurses TUI
│   ├── api_server.h/cpp  # OpenAI-compatible API server
│   └── cli_server.h/cpp  # CLI server (HTTP + tool execution)
├── backends/
│   ├── api.h/cpp         # API backend base
│   ├── openai.h/cpp      # OpenAI implementation
│   ├── anthropic.h/cpp   # Anthropic implementation
│   ├── gemini.h/cpp      # Google Gemini
│   ├── ollama.h/cpp      # Ollama
│   ├── llamacpp.h/cpp    # llama.cpp
│   ├── tensorrt.h/cpp    # TensorRT-LLM
│   └── cli_client.h/cpp  # Remote CLI server proxy
├── tools/
│   ├── tools.h/cpp       # Tool registry
│   └── *.cpp             # Tool implementations
└── docs/
    ├── architecture.md   # This file
    ├── backends.md       # Backend implementations
    ├── frontend.md       # Frontend implementations
    ├── server.md         # Server implementations
    ├── cliserver.md      # CLI server/client architecture
    └── *.md              # Other documentation
```
