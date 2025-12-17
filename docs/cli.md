# CLI Module Documentation

## Overview

The CLI module (`frontends/cli.cpp` / `frontends/cli.h`) implements the interactive command-line interface for Shepherd. It handles user input, manages the conversation loop, executes tool calls, and displays responses with proper formatting.

## Architecture (v2.14.0)

### Standalone CLI Frontend

As of v2.14.0, CLI is a standalone Frontend that handles its own input/output without depending on TerminalIO. This is part of the frontend refactor to make each frontend self-contained.

### Main Components

1. **CLI Class** - Frontend subclass for interactive mode
   - Inherits from `Frontend`
   - Owns `Tools` instance for tool management
   - Owns `Replxx` instance for line editing
   - Display helpers: `show_tool_call()`, `show_tool_result()`, `show_error()`
   - Output methods: `write_colored()`, `write_raw()`

2. **GenerationThread** - Async token generation
   - Background thread for LLM inference
   - Non-blocking main loop
   - EventCallback for streaming

3. **Input Thread** - Background input reading
   - Uses replxx for interactive line editing
   - Producer-consumer queue for async input
   - Handles piped input from stdin

### Key Data Flow

```
User Input (replxx/stdin)
    │
    ├── input_thread reads input
    │
    ▼
input_queue (thread-safe)
    │
    ▼
CLI::run() main loop
    │
    ├── Slash commands → handle_slash_commands()
    │
    └── User message
            │
            ▼
        GenerationThread
            │
            ├── session->add_message(callback)
            │        │
            │        ▼
            │    EventCallback
            │        │
            │        └── cli.write_colored(content, type)
            │
            └── Response with tool_calls
                    │
                    ▼
                Tool execution loop
                    │
                    ├── pending_tool_calls queue
                    │
                    └── execute_tool() → add_message(result)
```

## EventCallback Pattern (v2.14.0)

CLI provides a streaming callback when submitting requests:

```cpp
GenerationRequest req;
req.type = Message::USER;
req.content = user_input;

// Event callback - frontend handles all display
req.callback = [&cli](Message::Type type,
                      const std::string& content,
                      const std::string& tool_name,
                      const std::string& tool_call_id) -> bool {
    if (type == Message::TOOL_REQUEST) {
        // Queue tool for execution
        cli.pending_tool_calls.push({tool_name, content, tool_call_id});
        return true;
    }
    cli.write_colored(content, type);
    return true;  // Continue streaming
};

gen_thread.submit(req);
```

The callback receives typed events:
- `Message::ASSISTANT` - Text deltas
- `Message::TOOL_REQUEST` - Tool call detected by backend filtering
- `Message::TOOL_RESPONSE` - Tool result (if passed through)
- `Message::THINKING` - Thinking content (if show_thinking enabled)

## Input Handling (v2.14.0)

### Standalone Input in CLI

CLI handles its own input via an internal thread and replxx:

```
┌─────────────────────────────────────────┐
│              CLI                         │
│  ┌─────────────────┐                    │
│  │  input_loop()   │  replxx/getline    │
│  │  thread         │                    │
│  └────────┬────────┘                    │
│           │                             │
│           ▼                             │
│  ┌─────────────────┐                    │
│  │  input_queue    │  condition_variable │
│  └────────┬────────┘                    │
└───────────┼─────────────────────────────┘
            │
            ▼
    CLI::run() main loop
```

### Input Features

- **Line editing** - Via replxx library
- **History** - Up/Down arrows, saved to ~/.shepherd_history
- **Ctrl+D** - EOF (exit)
- **Piped input** - Automatic detection and handling

### Input Sources

1. **Interactive Mode** (TTY detected)
   - Uses replxx for line editing
   - History loaded/saved automatically

2. **Piped Mode** (stdin redirection)
   - Line-by-line reading via std::getline
   - Automatic EOF detection

## Tool Call Execution Flow

When the model generates a tool call:

1. **Detect via Callback** - TOOL_REQUEST event queues to pending_tool_calls
2. **Extract from Response** - Also check resp.tool_calls if not queued
3. **Display to User** - `show_tool_call(name, params)`
4. **Execute Tool** - `tools.execute(name, args)`
5. **Sanitize Result** - UTF-8 sanitization for JSON safety
6. **Truncate if Needed** - Scale to fit context window
7. **Send to Backend** - `add_message(TOOL_RESPONSE, result)`
8. **Continue Loop** - Backend processes and generates next response

### Tool Result Truncation

- **Reserved Space**: System + last user + last assistant messages
- **Scaling Function**: Adjusts based on context size
- **User Override**: `--truncate LIMIT` flag
- **Progressive Truncation**: "[TRUNCATED...]" marker

## CLI Output

### ANSI Color Codes

```cpp
const char* CLI::ansi_color(Message::Type type) {
    switch (type) {
        case Message::SYSTEM: return "\033[31m";       // Red
        case Message::USER: return "\033[32m";         // Green
        case Message::TOOL_REQUEST: return "\033[33m"; // Yellow
        case Message::TOOL_RESPONSE: return "\033[36m";// Cyan
        case Message::THINKING: return "\033[90m";     // Gray
        default: return "";
    }
}
```

### Color Detection

Colors are enabled when:
- Interactive mode (both stdin and stdout are TTY)
- TERM is not "dumb" or empty
- NO_COLOR environment variable is not set

## Slash Commands

Commands starting with `/`:

| Command | Description |
|---------|-------------|
| `/provider <name>` | Switch provider |
| `/model <name>` | Switch model |
| `/config` | Show configuration |
| `/tools` | List available tools |
| `/sched` | Show/control scheduler |
| `exit` / `quit` | Exit program |

## Files

- `frontends/cli.h` - CLI class declaration
- `frontends/cli.cpp` - Implementation (~800 lines)
- `generation_thread.h/cpp` - Async generation
- `ansi.h` - ANSI escape code definitions

## Configuration

CLI behavior influenced by:
- `config->warmup` - Warmup message before first input
- `config->truncate_limit` - Max tokens for tool results
- `--thinking` flag - Show thinking blocks
- `--colors` / `--no-colors` - Override color detection

## Class Members

### Public State
- `tools` - Tools instance for tool execution
- `eof_received` - EOF detected flag
- `generation_cancelled` - Generation was cancelled
- `interactive_mode` - Is stdin/stdout a TTY
- `colors_enabled` - Output with ANSI colors
- `piped_eof` - Piped input exhausted
- `pending_tool_calls` - Queue of tool calls from callback
- `input_queue` - Thread-safe input queue

### Public Methods
- `init()` - Initialize CLI, replxx, start input thread
- `run()` - Main loop
- `show_tool_call()` - Display tool invocation
- `show_tool_result()` - Display tool result
- `show_error()` - Display error message
- `show_cancelled()` - Display cancellation notice
- `write_colored()` - Write text with ANSI colors
- `write_raw()` - Write text without colors
- `add_input()` - Queue input for processing
- `has_pending_input()` - Check input queue
- `wait_for_input()` - Wait with timeout

## Version History

- **2.14.0** - Standalone CLI (no TerminalIO dependency), direct replxx usage
- **2.13.0** - Unified input (no InputReader), EventCallback pattern
- **2.7.0** - Producer-consumer input with InputReader
- **2.6.0** - Frontend abstraction
