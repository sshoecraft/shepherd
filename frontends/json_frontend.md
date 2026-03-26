# JSON Frontend

## Overview

Machine-readable JSON-lines frontend for stdin/stdout integration. Designed for pipe-based machine-to-machine communication (e.g., chatroom adapters, orchestration scripts, test harnesses).

## Architecture

Inherits directly from `Frontend` (not `Server`). Simplest frontend -- no threads, no terminal handling, no replxx, no ncurses, no HTTP, no scheduler.

- **Input**: Synchronous `std::getline` from stdin, one JSON object per line
- **Output**: One JSON object per line to stdout, flushed immediately after each line
- **Tool execution**: Local (same as CLI/TUI/CLIServer, not like APIServer which returns tools to client)

## Protocol

### Input Format

```json
{"type": "user", "content": "your message here"}
```

Only `type: "user"` is accepted. Content supports slash commands (e.g., `/clear`, `/provider`).

### Output Types

| Type | Description | Fields |
|------|-------------|--------|
| `text` | Assistant response chunk | `content` |
| `thinking` | Reasoning/thinking chunk | `content` |
| `tool_use` | Tool call initiated | `name`, `params`, `id` |
| `tool_result` | Tool execution result | `name`, `id`, `success`, `summary`, `error` (on failure) |
| `end_turn` | Turn complete | `turns`, `total_tokens`, `cost_usd` (if pricing available) |
| `error` | Error occurred | `message`, `error_type` (optional) |
| `system` | System message | `content` |

### Event Flow Per Turn

```
user input -> [text...] -> [tool_use -> tool_result]* -> [text...] -> end_turn
```

Tool calls trigger recursive generation. Multiple tool call cycles may occur before the final `end_turn`.

## Usage

```bash
# Interactive pipe mode
shepherd --json -p provider_name

# Single query mode
shepherd --json --prompt "hello" -p provider_name

# Piped input
echo '{"type":"user","content":"hello"}' | shepherd --json -p provider_name
```

## Command-line

`--json` flag selects this frontend. Overrides TUI auto-detection. Compatible with all standard flags (`--nomcp`, `--notools`, `--prompt`, `--max-tokens`, etc.).

## History

- v2.38.0: Initial implementation
