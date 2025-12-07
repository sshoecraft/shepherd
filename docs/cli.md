# CLI Module Documentation

## Overview

The CLI module (`cli.cpp` / `cli.h`) implements the interactive command-line interface for Shepherd. It handles user input, manages the conversation loop, executes tool calls, and displays responses with proper formatting and color coding.

## Architecture

### Main Components

1. **CLI Class** - Manages user interaction and output display
   - `show_tool_call()` - Displays tool invocations with cyan highlighting
   - `show_tool_result()` - Shows tool results (truncated for display)
   - `show_error()` - Displays errors in red
   - `show_cancelled()` - Shows cancellation message
   - State flags: `eof_received`, `generation_cancelled`

2. **run_cli()** - Main interaction loop function
   - Initializes TerminalIO with model-specific markers
   - Handles user input (stdin or piped)
   - Processes slash commands
   - Manages conversation flow with backend
   - Executes tool calls and handles results

### Key Data Flow

```
User Input → Sanitization → Backend Generation → Tool Call Detection
                                                         ↓
User Display ← Result Truncation ← Tool Execution ← Tool Call Extraction
```

## Tool Call Execution Flow

When the model generates a tool call, the CLI:

1. **Extract Tool Call** - Uses `extract_tool_call()` to parse from response
2. **Display to User** - Shows tool name and parameters via `show_tool_call()`
3. **Execute Tool** - Calls `execute_tool()` from tool system
4. **Sanitize Result** - **UTF-8 sanitization added to handle binary data**
5. **Truncate if Needed** - Scales result to fit context window
6. **Send to Backend** - Adds as `Message::TOOL` to session
7. **Continue Loop** - Backend processes result and generates next response

### Tool Result Truncation

The CLI implements intelligent truncation to prevent tool results from consuming the entire context:

- **Reserved Space**: System message + last user message + last assistant message
- **Scaling Function**: `calculate_truncation_scale()` adjusts based on context size
  - Larger contexts allow higher percentage for tool results
  - Smaller contexts more conservative (leaves room for response)
- **User Override**: `--truncate LIMIT` flag sets explicit token limit
- **Progressive Truncation**: If result too large, truncates with "[truncated...]" marker

## Input Handling

### Producer-Consumer Model (v2.7.0+)

Input handling uses a dedicated thread (InputReader) that feeds into a queue:

```
┌─────────────────┐     ┌──────────────────┐
│  InputReader    │────▶│  Input Queue     │
│  Thread         │     │  (TerminalIO)    │
└─────────────────┘     └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │   Main Thread    │
                        │   (CLI loop)     │
                        └──────────────────┘
```

This allows:
- Queuing prompts while processing continues
- Interrupting current processing when new input arrives
- Scheduler can inject prompts into the same queue

### Input Sources

1. **Interactive Mode** (TTY detected)
   - Uses `replxx` library for readline functionality
   - History support, arrow keys, Ctrl+C/Ctrl+D handling
   - InputReader thread handles replxx input

2. **Piped Mode** (stdin redirection)
   - Reads lines from stdin via InputReader thread
   - No interactive prompts
   - Automatic EOF detection

### Interrupt Mechanism

When new input arrives during processing:
1. Input is queued via `tio.add_input()`
2. Tool loop checks `tio.has_pending_input()`
3. If pending, sets `g_generation_cancelled = true`
4. Current generation stops, main loop proceeds to next input

### Input Sanitization

All user input is sanitized through `utf8_sanitizer::strip_control_characters()`:
- Removes terminal escape sequences
- Strips backspace, DEL, and other control chars
- Preserves newline, tab, carriage return
- Keeps valid UTF-8 sequences

## Terminal Output Filtering

The CLI configures TerminalIO with model-specific markers to hide:

1. **Tool Call Blocks** - XML/JSON tool invocations not shown to user
2. **Thinking Blocks** - Reasoning content (shown with `--thinking` flag)
3. **Special Tokens** - Backend-specific control sequences

Markers are:
- Provided by backend (from chat template)
- Fall back to common formats if not specified

## Slash Commands

The CLI supports special commands starting with `/`:

- **Provider Commands** - Handled by `handle_provider_command()`
  - `/switch <provider>` - Change active provider
  - `/list` - Show available providers
  - Other provider-specific commands

Commands not recognized are treated as regular user input.

## Recent Changes

### UTF-8 Sanitization for Tool Results (Current)

**Problem**: Binary data (e.g., JPEG images) returned from tool execution caused JSON parsing errors when sent to backend. Errors included:
- `[json.exception.type_error.316] invalid UTF-8 byte at index 0: 0xFF` (raw binary)
- `[json.exception.type_error.316] invalid UTF-8 byte at index 2: 0x3F` (broken sanitizer output)
- `[json.exception.type_error.316] invalid UTF-8 byte at index 435: 0xC0` (overlong encoding)

**Solution**: Three-part fix:

1. **Added UTF-8 sanitization after tool execution** (cli.cpp line 831):
```cpp
// Sanitize tool result to ensure valid UTF-8 for JSON
result_content = utf8_sanitizer::sanitize_utf8(result_content);
```

2. **Fixed replacement character bug in utf8_sanitizer.cpp**: The replacement character was incorrectly written as `\xEF\xBF?` instead of `\xEF\xBF\xBD`. The character `'?'` (0x3F) created invalid UTF-8 sequences, causing secondary errors. All four instances corrected to use `\xBD` (proper UTF-8 encoding for U+FFFD).

3. **Added validation for invalid UTF-8 bytes**: The sanitizer now rejects:
   - Overlong encodings (0xC0, 0xC1)
   - Invalid high bytes (0xF5-0xFF)

This ensures all tool results contain valid UTF-8 before being:
1. Displayed to user
2. Counted for token usage
3. Sent to backend via JSON API

Invalid UTF-8 sequences are now correctly replaced with Unicode replacement character (U+FFFD = 0xEF 0xBF 0xBD).

### Context Management

- Session handles message eviction when context full
- Delta method for accurate token counting
- EMA (Exponential Moving Average) for chars-per-token ratio

## Important Implementation Details

### Memory Safety
- Direct access to CLI state members (no getter functions per RULES.md)
- Session passed by reference, backend by unique_ptr

### Error Handling
- Tool execution failures shown as "Error: ..." messages
- Backend errors displayed via `show_error()`
- Generation cancellation handled gracefully

### Threading
- InputReader runs in dedicated thread for input handling
- Main CLI loop consumes from input queue
- Backend may use threads internally (e.g., streaming responses)
- Scheduler uses SIGALRM for timer-based prompt injection

## Dependencies

- `terminal_io.h` - Low-level terminal I/O with marker filtering and queue
- `input_reader.h` - Async input reading thread
- `tools/tool.h` - Tool execution system
- `tools/utf8_sanitizer.h` - UTF-8 validation and sanitization
- `backends/backend.h` - Backend interface
- `session.h` - Conversation session management
- `scheduler.h` - SIGALRM-based scheduled prompts
- `replxx` library - Interactive readline functionality

## Configuration

CLI behavior is influenced by:
- `config->warmup` - Send warmup message before first user input
- `config->truncate_limit` - Max tokens for tool results (0 = auto)
- `config->calibration` - Enable token count calibration
- `g_show_thinking` - Display thinking blocks (`--thinking` flag)
- `g_debug_level` - Debug logging verbosity

## Testing Notes

When testing tool execution:
- Binary data (images, PDFs) must be handled gracefully
- Large tool results should trigger truncation
- Tool execution errors should not crash CLI
- Context overflow should evict old messages, not fail
