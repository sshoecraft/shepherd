# Frontend Module Documentation

## Overview

The Frontend module defines the base class for all user-facing presentation layers in Shepherd. It provides a common interface for different interaction modes: CLI, API Server, and CLI Server.

## Architecture

### Class Hierarchy

```
Frontend (base class)
├── CLI          - Interactive command-line interface
├── Server       - Base class for server modes
│   ├── APIServer   - OpenAI-compatible HTTP API
│   └── CLIServer   - HTTP server with local tool execution
```

### Frontend Base Class

Located in `frontend.h` / `frontend.cpp`:

```cpp
class Frontend {
public:
    Frontend();
    virtual ~Frontend();

    static std::unique_ptr<Frontend> create(const std::string& mode,
                                            const std::string& host, int port);

    virtual int run(std::unique_ptr<Backend>& backend, Session& session) = 0;
};
```

### Factory Method

`Frontend::create()` returns the appropriate frontend based on mode:
- `"cli"` - Returns CLI instance
- `"api-server"` - Returns APIServer instance
- `"cli-server"` - Returns CLIServer instance

## Frontend Types

### CLI (`cli.h` / `cli.cpp`)

Interactive command-line interface for direct user interaction.

Features:
- readline support via replxx library
- Slash command processing (`/provider`, `/model`, `/config`, `/sched`, `/tools`)
- Tool execution and result display
- Context management with auto-eviction
- UTF-8 sanitization

### APIServer (`server/api_server.h` / `server/api_server.cpp`)

OpenAI-compatible HTTP API server for programmatic access.

Endpoints:
- `POST /v1/chat/completions` - Chat completion
- `GET /v1/models` - List models
- `GET /health` - Health check

### CLIServer (`server/cli_server.h` / `server/cli_server.cpp`)

HTTP server that executes tools locally while accepting remote prompts.

## Run Flow

1. `main.cpp` creates the appropriate Frontend via factory
2. Backend is created and initialized
3. Session is configured with tools and system prompt
4. `frontend->run(backend, session)` enters the main loop
5. Frontend handles user/client interaction until exit

## Files

- `frontend.h` / `frontend.cpp` - Base class
- `cli.h` / `cli.cpp` - CLI implementation
- `server/server.h` / `server/server.cpp` - Server base class
- `server/api_server.h` / `server/api_server.cpp` - API server
- `server/cli_server.h` / `server/cli_server.cpp` - CLI server

## Version History

- **2.6.0** - Added Frontend abstraction and factory pattern
- **2.5.0** - Original CLI-only implementation
