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

## Provider-as-Tool

When using API providers, non-active providers are registered as tools:
- If using OpenAI, `ask_anthropic` and `ask_google` become available
- Allows model to delegate to other providers for specialized tasks
- Managed via `register_provider_tools(active_provider)`

## History

- v2.6.0: Replaced ToolRegistry singleton with instance-based Tools class
  - CLI and CLIServer each own their Tools instance
  - Removed global execute_tool() function
  - Added tools command and /tools slash command with enable/disable support
