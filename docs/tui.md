# TUI Module

## Overview

The TUI class (`tui.cpp/h`) provides an NCurses-based visual interface. As of v2.13.0, TUI is **visual-only** - all input handling is done by TerminalIO.

## Architecture

```
┌─────────────────────────────────────────┐
│              Output Window              │  ← Scrollable output
│         (scrollback buffer)             │
├─────────────────────────────────────────┤
│           Pending Queue                 │  ← Queued inputs (gray/cyan)
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │ > input text here_                  │ │  ← Input box with cursor
│ └─────────────────────────────────────┘ │
├─────────────────────────────────────────┤
│ model: gpt-4    tokens: 1234/8192     │  ← Status line
└─────────────────────────────────────────┘
```

## NCurses Windows

| Window | Purpose | Properties |
|--------|---------|------------|
| output_win | Scrollable output | scrollok=TRUE |
| pending_win | Queued input display | Variable height |
| input_win | Input box with border | Fixed position |
| status_win | Status bar | Bottom row |

## Key Features

### Output Area
- Full scrollback buffer (default 10000 lines)
- Page up/down navigation (via TerminalIO)
- Auto-scroll to bottom on new content
- Line types with colors (USER, ASSISTANT, TOOL, etc.)

### Input Box
- Border with prompt "> "
- Multi-line support (box grows)
- Cursor positioning
- Updated by TerminalIO's input handler

### Queued Input Display
- Shows pending inputs during generation
- Gray: queued
- Cyan: currently processing

### Status Line
- Left: model info
- Right: token counts

## API

### Initialization

```cpp
TUI();
~TUI();
bool init();      // Initialize ncurses
void shutdown();  // Cleanup ncurses
```

### Output

```cpp
enum class LineType {
    USER,        // Green "> " prefix
    TOOL_CALL,   // Yellow "* " prefix
    TOOL_RESULT, // Cyan
    ASSISTANT,   // Default color
    SYSTEM       // Red/gray
};

void write_output(const std::string& text, LineType type);
void write_output(const char* text, size_t len, LineType type);
```

### Input Display (called by TerminalIO)

```cpp
void set_input_content(const std::string& content);
void position_cursor_in_input(int col);
std::string get_input_content() const;
void clear_input();
```

### Status

```cpp
void set_status(const std::string& left, const std::string& right);
```

### Event Loop

```cpp
bool run_once();  // Handle resize, refresh if needed
                  // Returns false if quit requested
```

### Queued Input

```cpp
void show_queued_input(const std::string& input);
void mark_input_processing();
void clear_queued_input_display();
```

### Control

```cpp
void request_quit();
void request_refresh();
void flush();  // Force immediate update
bool check_escape_pressed();
void set_escape_pressed();
```

## Visual-Only Design

TUI does NOT handle input directly:

```cpp
// OLD (removed):
// wgetch() in TUI
// process_input() in TUI

// NEW:
// TerminalIO::input_loop() handles all input
// TUI::set_input_content() updates display
// TUI::position_cursor_in_input() positions cursor
```

This separation means:
1. Same input behavior in console and TUI modes
2. TUI only needs to render what TerminalIO tells it
3. No duplicate input handling code

## Window Layout

```cpp
void calculate_layout() {
    // From bottom:
    status_height = 1;
    input_height = 3 + extra_lines;  // Min 3
    pending_height = queued_count;   // 0-2 lines
    output_height = term_rows - pending_height - input_height - status_height;
}
```

## Color Pairs

```cpp
enum ColorPairs {
    PAIR_DEFAULT = 0,
    PAIR_RED = 1,
    PAIR_GREEN = 2,
    PAIR_YELLOW = 3,
    PAIR_CYAN = 4,
    PAIR_GRAY = 5,
};
```

## Refresh Strategy

Uses double-buffering:
```cpp
wnoutrefresh(output_win);
wnoutrefresh(pending_win);
wnoutrefresh(input_win);
wnoutrefresh(status_win);
doupdate();  // Single screen update
```

## Global Instance

```cpp
extern TUI* g_tui;  // nullptr if not in TUI mode
```

Created by TerminalIO when `tui_mode=true`:
```cpp
// In TerminalIO::init():
if (tui_mode) {
    tui = new TUI();
    tui->init();
    g_tui = tui;
}
```

## Terminal Compatibility

- Disables alternate screen buffer (preserves scrollback)
- Disables mouse capture
- Uses standard color pairs
- Handles terminal resize via run_once()
