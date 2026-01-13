# Session Manager

## Overview

The `SessionManager` class provides multi-tenant stateful session support for the API server. It allows authenticated users with `permissions.server_tools = true` to have persistent server-side sessions with server-side tool execution.

## Architecture

### Key Components

1. **ManagedSession** - Per-API-key session state
   - `session` - Unique session for this user
   - `tools` - Tools instance with native tools registered
   - `session_mutex` - Per-session serialization
   - `last_access` - Timestamp for idle tracking
   - `requests_processed`, `tool_executions` - Statistics

2. **SessionManager** - Manages all sessions
   - Maps API keys to ManagedSession instances
   - Thread-safe with `manager_mutex`
   - Creates sessions on-demand

### Request Flow

```
Request arrives at /v1/chat/completions
    ↓
Extract API key from Authorization header
    ↓
If valid key with permissions.server_tools=true:
    → SessionManager.get_session(api_key)
    → handle_stateful_request()
    → Server-side tool execution
    → Stream response
Otherwise:
    → handle_stateless_request() (standard OpenAI behavior)
```

## Configuration

### Server Config

- `--auth-mode json` - Enable JSON file-based API key authentication (requires valid key)

### Per-Key Permissions

Edit `~/.config/shepherd/api_keys.json`:

```json
{
    "sk-abc123...": {
        "name": "internal-user",
        "permissions": {
            "server_tools": true    // Enables stateful session
        }
    },
    "sk-def456...": {
        "name": "external-app",
        "permissions": {
            "server_tools": false   // Standard OpenAI behavior
        }
    }
}
```

## Behavior Matrix

### --auth-mode none (default)

| API Key | Result |
|---------|--------|
| Any/None | Stateless, standard OpenAI behavior |

### --auth-mode json

| API Key | server_tools | Result |
|---------|--------------|--------|
| None | - | 401 API key required |
| Invalid | - | 401 Invalid API key |
| Valid | false/missing | Stateless, standard OpenAI behavior |
| Valid | true | Stateful session, server-side tools |

## Implementation Details

### Session Creation

Sessions are created lazily on first request:
1. Native tools registered (filesystem, command, json, http, memory, mcp_resource, core)
2. System message from config
3. Tool list populated in session

### Message Handling

In stateful mode:
- Only the **last user message** is extracted from the request
- Server session history is authoritative
- Client-provided history is ignored (prevents context drift)

### Tool Execution

When the model calls a tool:
1. `TOOL_CALL` event fires
2. Server executes tool via `Frontend::execute_tool()`
3. Result added to session as `TOOL_RESPONSE`
4. Generation continues automatically

### Session Lifecycle

- Sessions persist until server restart
- No automatic timeout/cleanup (can be added later)
- Manual clear via future `/v1/sessions/:id` endpoint

## Files

- `session_manager.h` - Class definitions
- `session_manager.cpp` - Implementation
- `frontends/api_server.cpp` - Integration with request routing

## Version History

- 2.19.0 - Initial implementation of multi-tenant stateful sessions
