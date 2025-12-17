# TerminalIO Module

## Overview

`TerminalIO` (terminal_io.cpp/h) is the unified input/output handling layer for terminal-based interfaces. It manages:
- Raw terminal mode and input processing
- Output with filtering (tool calls, thinking blocks)
- Writer functions for different output modes
- TUI integration

## Key Components

### Input Handling

As of v2.13.0, TerminalIO owns unified input processing for both console and TUI modes.

```cpp
class TerminalIO {
    // Input thread
    std::thread input_thread;
    std::atomic<bool> input_running{false};

    // Input state
    std::string input_content;
    int cursor_position{0};
    std::vector<std::string> history;
    int history_index{-1};
    bool in_paste_mode{false};

    // Methods
    void input_loop();           // Thread function
    void process_input(const char* buf, size_t len);
    void submit_input();
    void redraw_input_line();
};
```

### Input Features

| Feature | Key/Sequence | Description |
|---------|--------------|-------------|
| History up | Up arrow | Previous command |
| History down | Down arrow | Next command |
| Cursor left | Left arrow | Move cursor left |
| Cursor right | Right arrow | Move cursor right |
| Home | Home / Ctrl+A | Start of line |
| End | End / Ctrl+E | End of line |
| Delete | Delete | Delete char at cursor |
| Backspace | Backspace | Delete char before cursor |
| Clear line | Ctrl+U | Clear entire line |
| Kill to end | Ctrl+K | Delete to end of line |
| Cancel | Escape | Cancel current generation |
| Clear input | Escape Escape | Double-escape clears input |
| Quit | Ctrl+D | Exit (on empty line) |
| Paste | Bracketed paste | Multi-line paste support |

### Input Flow

```
input_loop() [thread]
    │
    ├─→ poll(stdin, 50ms timeout)
    │
    ├─→ read(buf, 256)
    │
    └─→ process_input(buf, len)
        │
        ├─→ Escape sequences → cursor movement, history
        ├─→ Ctrl keys → line editing
        ├─→ Enter → submit_input()
        └─→ Regular chars → insert at cursor
```

### Bracketed Paste

Terminal bracketed paste mode is enabled:
- Start: `\033[?2004h`
- End: `\033[?2004l`
- Paste start: `\033[200~`
- Paste end: `\033[201~`

In paste mode, Enter adds newline instead of submitting.

## Output Handling

### Writer Functions

TerminalIO uses pluggable writer functions:

```cpp
using WriterFunc = void (*)(const char* text, size_t len, Message::Type type);

// Built-in writers:
static void console_writer(...);  // Direct stdout with colors
static void tui_writer(...);      // Route to TUI windows
static void server_writer(...);   // Timestamps + type prefixes
```

Set via `tio.set_writer(func)`.

### Output Filtering

The `write()` method filters output:
- Tool call markers hidden during streaming
- Thinking blocks optionally hidden
- Formatted tool call display after detection

Filter state machine:
```
NORMAL → DETECTING_TAG → IN_TOOL_CALL → CHECKING_CLOSE → NORMAL
                       → IN_THINKING  → CHECKING_CLOSE → NORMAL
```

### Color Support

```cpp
enum class Color {
    DEFAULT,
    RED,
    YELLOW,
    GREEN,
    CYAN,
    GRAY
};

// Usage
tio.write(text, Message::USER);     // Green
tio.write(text, Message::ASSISTANT); // Default
tio.write(text, Message::SYSTEM);    // Red
```

## Output Markers

Tool and thinking markers are model-specific:

```cpp
struct OutputMarkers {
    std::vector<std::string> tool_call_start;   // e.g., "<tool_call>"
    std::vector<std::string> tool_call_end;     // e.g., "</tool_call>"
    std::vector<std::string> thinking_start;    // e.g., "<think>"
    std::vector<std::string> thinking_end;      // e.g., "</think>"
};

tio.markers = backend->get_markers();
```

## TUI Integration

When in TUI mode (`tio.tui_mode == true`):
- `tui_writer` routes output to TUI windows
- Input thread updates TUI's input display
- TUI handles visual rendering only

```cpp
if (tui_mode && tui) {
    tui->set_input_content(input_content);
    tui->position_cursor_in_input(cursor_position);
}
```

## Public Interface

### Initialization

```cpp
bool init(bool interactive, bool colors, bool tui_mode);
void shutdown();
```

### Input

```cpp
std::string read(const char* prompt);
void add_input(const std::string& text, bool needs_echo = false);
void clear_input_queue();
bool check_escape_pressed();
```

### Output

```cpp
void write(const char* text, size_t len, Message::Type type);
void write(const std::string& text, Message::Type type);
void error(const std::string& msg);
void debug(int level, const std::string& msg);
```

### Generation Control

```cpp
void reset();            // Reset filter state
void begin_response();   // Start generation
void end_response();     // End generation
std::atomic<bool> is_generating{false};
```

## Global Instance

```cpp
extern TerminalIO tio;  // Single global instance
```

All code uses `tio.write()`, `tio.read()`, etc.

## Thread Safety

- Input queue protected by mutex + condvar
- Output writes are not thread-safe (single writer assumed)
- `is_generating` is atomic for cross-thread status

## Error Handling

```cpp
tio.error("Something went wrong");  // Red output
tio.debug(1, "Debug info");         // Conditional on g_debug_level
```
