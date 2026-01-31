# MCP (Model Context Protocol) Module

## Overview
The MCP module manages connections to external MCP and SMCP (Secure MCP) servers, discovering and registering their tools with Shepherd.

## Architecture

### Components
- **MCP** (mcp/mcp.cpp, mcp/mcp.h): Singleton manager for all MCP server connections
- **MCPServer** (mcp/mcp_server.cpp, mcp/mcp_server.h): Handles individual server process lifecycle
- **MCPClient** (mcp/mcp_client.cpp, mcp/mcp_client.h): JSON-RPC 2.0 protocol client
- **MCPToolAdapter** (mcp/mcp_tool.h): Adapts MCP tools to Shepherd's Tool interface

### Initialization Flow
```
MCP::initialize()
    |
    +-- Parse mcp_config and smcp_config JSON
    |
    +-- For each server (in parallel):
    |       |
    |       +-- MCPServer::start()
    |       |       |-- fork()
    |       |       |-- prctl(PR_SET_PDEATHSIG, SIGTERM)  // child auto-dies with parent
    |       |       |-- execvp(command)
    |       |       +-- SMCP handshake if credentials present
    |       |
    |       +-- MCPClient::initialize()
    |       |       |-- JSON-RPC initialize request
    |       |       +-- notifications/initialized
    |       |
    |       +-- MCPClient::list_tools()
    |
    +-- Register tools (single-threaded)
            |-- Create MCPToolAdapter for each tool
            +-- tools.register_tool()
```

### Thread Safety
- **init_server()**: Thread-safe, creates independent objects, no shared state
- **register_server()**: Single-threaded, modifies clients_, servers_by_name_, total_tools_
- **Tool calls**: Thread-safe via MCPClient per-server isolation

### Process Management
- Child processes receive `SIGTERM` via `PR_SET_PDEATHSIG` when parent exits
- Shutdown is non-blocking - just sends SIGTERM and closes file descriptors

## Configuration

### MCP Servers (mcp_config)
```json
[
  {
    "name": "server-name",
    "command": "/path/to/server",
    "args": ["--arg1", "value1"],
    "env": {"KEY": "value"}
  }
]
```

### SMCP Servers (smcp_config)
```json
[
  {
    "name": "server-name",
    "command": "/path/to/server",
    "args": [],
    "credentials": {"key": "secret"}
  }
]
```

## History
- v2.25.0: Parallelized server initialization, simplified shutdown with PR_SET_PDEATHSIG
- Initial: Sequential server initialization
