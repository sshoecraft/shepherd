# InputReader Module Documentation

## Overview

The InputReader module provides asynchronous input handling for Shepherd. It runs a dedicated thread that reads user input from the terminal (via replxx) or stdin (piped mode) and queues it for processing by the main loop.

## Architecture

### Thread Model

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

### Key Components

- `input_reader.h` - Class declaration
- `input_reader.cpp` - Implementation

## Class: InputReader

### Public Members

- `Replxx* replxx` - The replxx instance (public for TerminalIO output access)

### Methods

#### Initialization

```cpp
bool init(bool interactive, bool colors_enabled, InputCallback callback);
```

Initializes the reader:
- `interactive`: true for replxx (TTY), false for stdin (piped)
- `colors_enabled`: whether to use colored prompts
- `callback`: function called when input is received (typically queues to TerminalIO)

#### Lifecycle

```cpp
void start();   // Start the reader thread
void stop();    // Stop the reader thread gracefully
bool is_running() const;  // Check if running
```

#### History Management

```cpp
void history_add(const std::string& line);
void history_load(const std::string& path);
void history_save(const std::string& path);
```

## Integration with TerminalIO

The InputReader feeds into TerminalIO's input queue:

1. InputReader reads a line from user/stdin
2. Calls the callback function with the input
3. Callback typically does `tio.add_input(input)`
4. Main loop waits on `tio.read()` which pulls from queue

### EOF Handling

When EOF is detected (Ctrl+D or end of piped input):
1. InputReader calls callback with empty string
2. Callback signals EOF to main loop
3. Main loop breaks out gracefully

## Usage in CLI

```cpp
// Create reader
InputReader input_reader;
std::atomic<bool> eof_signaled{false};

// Setup callback
auto callback = [&eof_signaled](const std::string& input) {
    if (input.empty()) {
        eof_signaled = true;
        tio.notify_input();
    } else {
        tio.add_input(input);
    }
};

// Initialize and start
input_reader.init(tio.interactive_mode, tio.colors_enabled, callback);
input_reader.start();

// Main loop consumes from tio.read()
while (true) {
    std::string input = tio.read(">");
    if (input.empty() && eof_signaled) break;
    // Process input...
}

// Cleanup
input_reader.stop();
```

## Interrupt Support

When new input arrives while the main loop is processing:
1. Input is queued via `tio.add_input()`
2. Main loop can check `tio.has_pending_input()` during processing
3. If pending input exists, main loop can set `g_generation_cancelled = true`
4. Current generation stops, main loop proceeds to next queued input

## Files

- `input_reader.h` - Class declaration
- `input_reader.cpp` - Implementation
- `terminal_io.h/cpp` - Queue infrastructure (condition variable, mutex)
- `cli.cpp` - Integration point

## History

- v2.7.0: Initial implementation - producer-consumer input queue model
