# Frontend Module

## Overview

The Frontend class (`frontend.cpp/h`) is the base class for all presentation layers. It manages providers, backend lifecycle, and common initialization.

## Class Hierarchy

```
Frontend (abstract base)
    │
    ├── CLI
    │   └── Interactive terminal mode
    │
    └── Server (abstract)
        │
        ├── CLIServer
        │   └── HTTP server with tool execution
        │
        └── APIServer
            └── OpenAI-compatible API
```

## Frontend Base Class

```cpp
class Frontend {
public:
    // Factory method
    static std::unique_ptr<Frontend> create(
        const std::string& mode,      // "cli", "api-server", "cli-server"
        const std::string& host,
        int port,
        Session& session,
        Provider* cmdline_provider = nullptr,
        bool no_mcp = false,
        bool no_tools = false
    );

    // Initialization
    virtual void init(Session& session, bool no_mcp, bool no_tools) {}

    // Main loop - subclasses implement
    virtual int run(Session& session) = 0;

    // Provider management
    Provider* get_provider(const std::string& name);
    std::vector<std::string> list_providers() const;
    bool connect_next_provider(Session& session);
    bool connect_provider(const std::string& name, Session& session);

    // State
    std::vector<Provider> providers;
    std::string current_provider;
    std::unique_ptr<Backend> backend;

protected:
    // Common tool initialization
    static void init_tools(Session& session, Tools& tools, bool no_mcp, bool no_tools);
};
```

## Factory Method

`Frontend::create()` handles:
1. Creates appropriate frontend subclass
2. Loads providers from disk
3. Adds command-line provider (if any)
4. Sets output writer
5. Calls `init()` for tool setup

```cpp
auto frontend = Frontend::create("cli", "", 0, session, nullptr, false, false);
frontend->connect_next_provider(session);
return frontend->run(session);
```

## Common Tool Initialization

`init_tools()` is shared by CLI and CLIServer (v2.13.0):

```cpp
void Frontend::init_tools(Session& session, Tools& tools, bool no_mcp, bool no_tools) {
    // 1. Initialize RAG system
    RAGManager::initialize(db_path, config->max_db_size);

    if (no_tools) return;

    // 2. Register native tools
    register_filesystem_tools(tools);
    register_command_tools(tools);
    register_json_tools(tools);
    register_http_tools(tools);
    register_memory_tools(tools);
    register_mcp_resource_tools(tools);
    register_core_tools(tools);

    // 3. Initialize MCP servers
    if (!no_mcp) {
        MCP::instance().initialize(tools);
    }

    // 4. Build combined tool list
    tools.build_all_tools();

    // 5. Populate session
    tools.populate_session_tools(session);
}
```

## Provider System

### Provider Structure

```cpp
struct Provider {
    std::string name;
    std::string backend;     // "openai", "anthropic", etc.
    std::string model;
    std::string api_key;
    std::string api_base;
    int priority;            // Lower = higher priority
};
```

### Connection Flow

```cpp
bool connect_next_provider(Session& session) {
    for (auto& provider : providers) {
        try {
            backend = provider.connect(session);
            if (backend) {
                current_provider = provider.name;
                session.backend = backend.get();
                return true;
            }
        } catch (...) {
            // Try next provider
        }
    }
    return false;
}
```

## CLI Frontend

Interactive terminal mode with tool execution.

```cpp
class CLI : public Frontend {
public:
    void init(Session& session, bool no_mcp, bool no_tools) override {
        Frontend::init_tools(session, tools, no_mcp, no_tools);
    }

    int run(Session& session) override;

    // Tool management
    Tools tools;

    // Output helpers
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& result);
    void show_error(const std::string& error);
    void show_cancelled();

    // Slash commands
    bool handle_slash_commands(const std::string& input, Session& session);
};
```

### CLI Run Loop

```cpp
int CLI::run(Session& session) {
    GenerationThread gen_thread;
    gen_thread.init(&session);
    gen_thread.start();

    while (true) {
        // 1. Get user input
        auto [input, needs_echo] = tio.read_with_echo_flag(">");

        // 2. Handle slash commands
        if (handle_slash_commands(input, session)) continue;

        // 3. Submit to generation thread with callback
        GenerationRequest req;
        req.type = Message::USER;
        req.content = input;
        req.callback = [](Message::Type type, const std::string& content, ...) {
            tio.write(content, type);
            return true;
        };
        gen_thread.submit(req);

        // 4. Wait for completion
        while (!gen_thread.is_complete()) {
            // Poll TUI, check escape, etc.
        }

        // 5. Handle tool calls
        while (has_tool_calls(resp)) {
            execute_tool();
            submit_result();
        }
    }
}
```

## Output Writers

Frontend mode determines the output writer:

| Mode | Writer | Description |
|------|--------|-------------|
| cli (console) | console_writer | Direct stdout with ANSI colors |
| cli (tui) | tui_writer | Route to TUI windows |
| api-server | server_writer | Timestamps + type prefixes |
| cli-server | server_writer | Timestamps + type prefixes |

## Files

- `frontend.h` / `frontend.cpp` - Base class and init_tools()
- `cli.h` / `cli.cpp` - CLI implementation
- `server/server.h` / `server/server.cpp` - Server base class
- `server/api_server.h` / `server/api_server.cpp` - API server
- `server/cli_server.h` / `server/cli_server.cpp` - CLI server

## Version History

- **2.13.0** - Added init_tools() to eliminate CLI/CLIServer duplication
- **2.6.0** - Added Frontend abstraction and factory pattern
- **2.5.0** - Original CLI-only implementation
