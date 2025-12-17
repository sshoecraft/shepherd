● Documentation complete. Here's the final summary:

  Implementation Complete (v2.13.0)

  Code Changes

  Part 1: Frontend/Backend Refactor
  - Unified add_message() with optional EventCallback parameter
  - Deleted add_message_stream() from all backends and Session
  - EventCallback signature: bool(Message::Type, content, tool_name, tool_call_id)
  - Backends never output directly - only via callback
  - Updated: all backends, session, generation_thread, cli, servers

  Part 2: Unified Input
  - Renamed TUIScreen → TUI (tui.h, tui.cpp)
  - Unified input in TerminalIO with process_input()
  - Deleted InputReader (input_reader.cpp, input_reader.h)
  - Input features: double-escape, Ctrl+U/K/A/E, history, bracketed paste
  - TUI now ncurses visual-only

  Part 3: Code Deduplication
  - Extracted common tool init to Frontend::init_tools()
  - CLI::init() and CLIServer::init() call shared function

  Documentation Updated

  | File                      | Description                        |
  |---------------------------|------------------------------------|
  | docs/architecture.md      | Core architecture with event model |
  | docs/backends.md          | Backend module with EventCallback  |
  | docs/terminal_io.md       | Unified input/output handling      |
  | docs/tui.md               | NCurses visual-only TUI            |
  | docs/frontend.md          | Frontend base class                |
  | docs/session.md           | Session with EventCallback         |
  | docs/generation_thread.md | Async generation                   |
  | docs/server.md            | Server module overview             |
  | docs/cli.md               | CLI module (updated for v2.13.0)   |
  | docs/cli_server.md        | CLI server (updated for v2.13.0)   |
  | docs/api_server.md        | API server (updated for v2.13.0)   |
  | docs/output.md            | Output merged into TerminalIO      |

● Build successful. Here's a summary of the changes made:

  Summary of Fixes

  Problem: Double prompt display - user input was being shown twice (once from input line, once from echo function)

  Root Cause: The architecture should be event-driven where the backend fires a Message::USER callback when the provider accepts the prompt, but:
  1. No backends were firing the USER event
  2. Frontend was manually echoing the prompt before generation

  Changes Made:

  1. terminal_io.cpp - submit_input():
    - Clear input line properly in console mode
    - Don't redraw prompt (it will come from USER event)
  2. terminal_io.cpp - write():
    - Added USER message formatting with "> " prefix and proper indentation
  3. terminal_io.cpp - Added show_input_prompt():
    - Public method to redraw prompt after response completes
  4. cli.cpp:
    - Removed tio.echo_user_input() call
    - Added tio.show_input_prompt() after response/error/cancel
  5. All backends - Added USER callback firing:
    - api.cpp (base class for non-streaming)
    - anthropic.cpp
    - openai.cpp
    - gemini.cpp
    - ollama.cpp
    - tensorrt.cpp
    - llamacpp.cpp
  6. CLI server/client (for remote connections):
    - cli_server.cpp - Handle USER type in stream_callback, send user_echo event
    - cli_client.cpp - Process user_echo events and fire callback with USER type

  Now the prompt displays via the event system when the provider accepts it, not prematurely from the frontend.



