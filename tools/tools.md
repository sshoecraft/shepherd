# Tools System Architecture

## Overview

The Tools system manages all tool registration, lookup, enable/disable, and execution in Shepherd. It replaces the former ToolRegistry singleton with a proper instance-based design where each frontend (CLI, CLIServer) owns its own Tools instance.

## Key Components

### Tools Class (`tools.h`, `tools.cpp`)

The main class that owns and manages all registered tools.

**Storage:**
- `core_tools` - Built-in tools (bash, grep, glob, edit, read, write, etc.)
- `mcp_tools` - Tools from MCP servers
- `api_tools` - Tools representing other API providers (ask_anthropic, etc.)
- `all_tools` - Flattened vector of raw pointers for iteration
- `by_name` - Map for O(1) lookup by name
- `enabled` - Map tracking which tools are enabled/disabled

**Key Methods:**
- `register_tool(unique_ptr<Tool>, category)` - Register a tool to a category
- `build_all_tools()` - Rebuild flat list and lookup map after registration
- `get(name)` - Look up tool by name
- `execute(name, params)` - Execute a tool and return ToolResult
- `populate_session_tools(session)` - Populate Session::Tool vector for backends
- `handle_tools_args(args)` - Handle `shepherd tools` and `/tools` commands

### Tool Base Class (`tool.h`)

Abstract base class all tools must implement:
- `unsanitized_name()` - Original name (may contain colons, etc.)
- `description()` - Tool description
- `parameters()` - Legacy string format for parameters
- `get_parameters_schema()` - Structured ParameterDef vector for JSON schema
- `execute(args)` - Execute with arguments, return result map

### Tool Categories

**Core Tools** (`core_tools.cpp`):
- BashTool, GlobTool, GrepTool, EditTool
- WebFetchTool, WebSearchTool
- TodoWriteTool, BashOutputTool, KillShellTool
- GetTimeTool, GetDateTool

**Filesystem Tools** (`filesystem_tools.cpp`):
- ReadFileTool, WriteFileTool, ListDirectoryTool

**Memory Tools** (`memory_tools.cpp`):
- SearchMemoryTool, StoreMemoryTool, ClearMemoryTool
- SetFactTool, GetFactTool, ClearFactTool

**Command Tools** (`command_tools.cpp`):
- ExecuteCommandTool, GetEnvironmentVariableTool, ListProcessesTool

**JSON Tools** (`json_tools.cpp`):
- ParseJSONTool, SerializeJSONTool, QueryJSONTool

**HTTP Tools** (`http_tools.cpp`):
- HTTPRequestTool, HTTPGetTool, HTTPPostTool

**MCP Resource Tools** (`mcp_resource_tools.cpp`):
- ListMcpResourcesTool, ReadMcpResourcesTool

## Registration Flow

1. Frontend creates Tools instance (CLI::tools or CLIServer::tools)
2. Frontend::init() calls register_*_tools(tools) for each category
3. MCP::initialize(tools) registers MCP server tools
4. APITools::initialize(tools) registers API tools
5. tools.build_all_tools() finalizes registration
6. tools.populate_session_tools(session) populates Session for backends

## Tool Execution

Tools are executed via `tools.execute(name, params)`:
1. Look up tool by name (case-insensitive)
2. Check if tool is enabled
3. Call tool->execute(params)
4. Convert result map to ToolResult (check for content/output/error keys)

## Enable/Disable

Tools can be enabled/disabled via:
- `/tools enable <name1> [name2] ...`
- `/tools disable <name1> [name2] ...`
- `shepherd tools enable/disable ...` (command line)

Disabled tools are excluded from session.tools and will return an error if executed.

## Provider-as-Tool (API Tools)

When using API providers, non-active providers are registered as tools (e.g., `ask_anthropic`, `ask_flash`). This enables multi-model collaboration where one model can delegate to another for specialized tasks.

### Architecture

**APIToolAdapter** (`api_tools.h`, `api_tools.cpp`) wraps an API provider as a callable tool:

```
APIToolAdapter
├── Provider provider          # Provider config (type, model, API key, etc.)
├── Session tool_session       # Dedicated session for this tool
├── unique_ptr<Backend> backend # Persistent backend (created once)
├── connected                   # Lazy initialization flag
└── callback state              # Accumulated content, pending tool calls
```

### Design Principles

**Mirrors CLI Pattern**: Uses `Provider.connect()` - the same flow as the main CLI:
- No global config manipulation
- Backend created once on first use, reused for subsequent calls
- Each tool has its own Session with proper sampling parameters

**Lazy Initialization**: Backend is created on first `execute()` call via `ensure_connected()`:
1. Creates callback that captures CONTENT, TOOL_CALL, ERROR events
2. Disables streaming (sub-backends must complete before returning)
3. Calls `Provider.connect(tool_session, callback)`
4. Backend persists across multiple calls

**Stateless Sessions**: After each call, the session is cleared:
```cpp
tool_session.messages.clear();
tool_session.total_tokens = 0;
```

### Nested Tool Execution

API tools can themselves use tools. When `ask_flash` calls `get_time`:

```
ask_flash(prompt="what time is it")
  get_time({})
    04:30:14
  The current time is 04:30:14.
```

The tool loop:
1. Add user message to `tool_session`
2. Call `backend->generate_from_session()`
3. If TOOL_CALL events received, execute those tools
4. Add tool results to session
5. Loop until no more tool calls (max 10 iterations)
6. Return accumulated content

### Registration

```cpp
// Register all non-active providers as tools
register_provider_tools(tools, active_provider_name);

// Register single provider
register_provider_as_tool(tools, "anthropic");

// Unregister
unregister_provider_tool(tools, "anthropic");
```

When switching providers (`/provider use X`), tools are automatically re-registered to exclude the new active provider.

### Tool Parameters

Each API tool accepts:
- `prompt` (required): The question or request
- `max_tokens` (optional): Maximum tokens to generate

### Example Usage

```
> use the ask_flash tool "what's the capital of France"

ask_flash(prompt=what's the capital of France)
  The capital of France is Paris.
```

With nested tools:
```
> use the ask_openai tool "what time is it"

ask_openai(prompt=what time is it)
  get_time({})
    14:30:25
  The current time is 14:30:25.
```

## History

- v2.7.0: Refactored APIToolAdapter to mirror CLI pattern
  - Uses Provider.connect() instead of global config swapping
  - Backend created once on first use (lazy initialization)
  - Each API tool has its own Session (thread-safe, no shared state)
  - Added nested tool call display with indentation
  - Fixed Gemini callback to emit TOOL_CALL events

- v2.6.0: Replaced ToolRegistry singleton with instance-based Tools class
  - CLI and CLIServer each own their Tools instance
  - Removed global execute_tool() function
  - Added tools command and /tools slash command with enable/disable support
