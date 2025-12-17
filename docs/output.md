# Output Module Documentation

## Overview

The Output module is now merged into TerminalIO (`terminal_io.cpp/h`). As of v2.13.0, TerminalIO handles both input and output for all terminal-based interfaces.

## TerminalIO Output Architecture

### Writer Functions

TerminalIO uses pluggable writer functions based on mode:

```cpp
using WriterFunc = void(*)(const char* text, size_t len, Message::Type type);

class TerminalIO {
    WriterFunc writer;  // Current writer function

    // Static writers
    static void console_writer(const char*, size_t, Message::Type);
    static void tui_writer(const char*, size_t, Message::Type);
    static void server_writer(const char*, size_t, Message::Type);
};
```

### Writer Selection

Frontend mode determines the writer:

| Mode | Writer | Description |
|------|--------|-------------|
| CLI (console) | `console_writer` | Direct stdout with ANSI colors |
| CLI (TUI) | `tui_writer` | Routes to NCurses windows |
| API Server | `server_writer` | Timestamps + type prefixes |
| CLI Server | `server_writer` | Timestamps + type prefixes |

### Message Type Rendering

| Message::Type | Console Color | Prefix |
|---------------|---------------|--------|
| USER | Green | `> ` |
| ASSISTANT | Default | (none) |
| TOOL_REQUEST | Yellow | `* ` |
| TOOL_RESPONSE | Cyan | (none) |
| SYSTEM | Red/Gray | (none) |

## Output Methods

### Main write() Methods

```cpp
void TerminalIO::write(const char* text, size_t len, Message::Type type);
void TerminalIO::write(const std::string& text, Message::Type type);
```

Delegates to current writer function with filtering applied.

### Filtered Output

TerminalIO has a state machine for filtering:
- Tool call blocks (XML/JSON markers)
- Thinking blocks (`--thinking` to show)

```cpp
enum FilterState { NORMAL, DETECTING_TAG, IN_TAG, IN_THINKING };
FilterState filter_state;

void TerminalIO::write(...) {
    // Apply filtering based on state machine
    for (char c : text) {
        switch (filter_state) {
        case NORMAL:
            if (c matches marker_start) {
                filter_state = DETECTING_TAG;
            } else {
                writer(c);
            }
            break;
        // ... handle other states
        }
    }
}
```

### Debug and Error Output

```cpp
void TerminalIO::debug(int level, const std::string& text);
void TerminalIO::error(const std::string& text);
```

- **debug(level, text)**: Outputs if `g_debug_level >= level`
- **error(text)**: Always outputs

Format: `[YYYY-MM-DD HH:MM:SS.mmm] [DEBUG|ERROR] text`

These write directly to stderr, bypassing the writer and filtering.

## Console Writer

Writes to stdout with ANSI colors:

```cpp
void TerminalIO::console_writer(const char* text, size_t len, Message::Type type) {
    if (colors_enabled) {
        switch (type) {
        case Message::USER:
            std::cout << "\033[32m";  // Green
            break;
        case Message::TOOL_REQUEST:
            std::cout << "\033[33m";  // Yellow
            break;
        // ...
        }
    }
    std::cout.write(text, len);
    if (colors_enabled) {
        std::cout << "\033[0m";
    }
    std::cout.flush();
}
```

## TUI Writer

Routes to NCurses windows via TUI class:

```cpp
void TerminalIO::tui_writer(const char* text, size_t len, Message::Type type) {
    if (!tio.tui) {
        console_writer(text, len, type);
        return;
    }

    // Map to TUI::LineType
    TUI::LineType line_type;
    switch (type) {
        case Message::USER:      line_type = TUI::LineType::USER; break;
        case Message::TOOL_REQUEST:
        case Message::TOOL_RESPONSE: line_type = TUI::LineType::TOOL_RESULT; break;
        case Message::ASSISTANT: line_type = TUI::LineType::ASSISTANT; break;
        default:                 line_type = TUI::LineType::SYSTEM; break;
    }

    tio.tui->write_output(text, len, line_type);
    tio.tui->flush();
}
```

## Server Writer

Adds timestamps and type prefixes:

```cpp
void TerminalIO::server_writer(const char* text, size_t len, Message::Type type) {
    // Get timestamp
    auto timestamp = format_timestamp();

    // Get type prefix
    const char* prefix = "";
    switch (type) {
        case Message::USER:     prefix = "[USER] "; break;
        case Message::ASSISTANT: prefix = "[ASST] "; break;
        // ...
    }

    std::cout << timestamp << " " << prefix;
    std::cout.write(text, len);
    std::cout.flush();
}
```

## Thread Safety

Writers are called from GenerationThread (background). Each writer handles its own synchronization:

- **console_writer**: Uses stdout which is thread-safe with flush
- **tui_writer**: TUI methods have internal mutex
- **server_writer**: Uses stdout with flush

The `debug()` and `error()` methods write to stderr which is thread-safe for line-buffered output.

## Data Flow (v2.13.0)

```
Backend generates token
    │
    ▼
EventCallback(type, content, ...)
    │
    ▼
tio.write(content, type)
    │
    ▼
Filter state machine
    │
    ├── Normal text → writer()
    │
    ├── Tool call marker → buffer until end
    │
    └── Thinking block → buffer or forward based on --thinking
            │
            ▼
        Writer function
            │
            ├── Console: stdout + ANSI colors
            ├── TUI: NCurses windows
            └── Server: stdout + timestamps
```

## Response Lifecycle

```cpp
// Before generation
tio.begin_response();  // Reset filter state

// During generation (via callback)
tio.write(delta, Message::ASSISTANT);  // Filtered output

// After generation
tio.end_response();  // Flush any pending content
```

## Files

- `terminal_io.h` - TerminalIO class with writer functions
- `terminal_io.cpp` - Implementation including writers
- `tui.h/cpp` - TUI class for NCurses output

## Version History

- **2.13.0** - Output merged into TerminalIO, EventCallback pattern
- **2.9.0** - Output module with debug()/error() methods
- **2.8.0** - Initial Output module (replaced g_output_queue)
