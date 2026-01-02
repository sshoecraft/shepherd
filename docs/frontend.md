# Frontend Module

## Overview

The Frontend class (`frontend.cpp/h`) is the base class for all presentation layers. It manages providers, backend lifecycle, and common initialization.

## Class Hierarchy

```
Frontend (abstract base)
    |
    +-- CLI
    |   +-- Interactive terminal mode
    |
    +-- Server (abstract)
        |
        +-- CLIServer
        |   +-- HTTP server with tool execution
        |   +-- See docs/cliserver.md
        |
        +-- APIServer
            +-- OpenAI-compatible API
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
        Provider* cmdline_provider = nullptr,
        bool no_mcp = false,
        bool no_tools = false
    );

    // Initialization
    virtual void init(bool no_mcp, bool no_tools) {}

    // Main loop - subclasses implement
    virtual int run(Provider* cmdline_provider = nullptr) = 0;

    // Event callback - subclasses can override for custom handling
    virtual bool on_event(CallbackEvent event,
                          const std::string& content,
                          const std::string& name,
                          const std::string& id) { return true; }

    // Provider management
    Provider* get_provider(const std::string& name);
    std::vector<std::string> list_providers() const;
    bool connect_next_provider();
    bool connect_provider(const std::string& name);

    // Tool execution (for CLI, TUI, CLIServer - NOT APIServer)
    ToolResult execute_tool(Tools& tools,
                           const std::string& tool_name,
                           const std::map<std::string, std::any>& parameters,
                           const std::string& tool_call_id);

    // State
    Session session;                      // Conversation state
    std::vector<Provider> providers;      // Available providers
    std::string current_provider;         // Currently connected provider
    std::unique_ptr<Backend> backend;     // Active backend
    Backend::EventCallback callback;      // Streaming callback
};
```

## Factory Method

`Frontend::create()` handles:
1. Creates appropriate frontend subclass
2. Loads providers from disk
3. Adds command-line provider (if any)
4. Calls `init()` for tool setup

```cpp
auto frontend = Frontend::create("cli", "", 0, nullptr, false, false);
frontend->run(cmdline_provider);
```

## Common Tool Initialization

`init_tools()` is shared by CLI and CLIServer:

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
    std::string backend;     // "openai", "anthropic", "llamacpp", etc.
    std::string model;
    std::string api_key;
    std::string api_base;
    int priority;            // Lower = higher priority
};
```

### Connection Flow

```cpp
bool connect_provider(const std::string& name) {
    Provider* provider = get_provider(name);
    if (!provider) return false;

    // Create backend based on provider type
    backend = create_backend_for_provider(*provider, session, callback);
    if (!backend) return false;

    current_provider = name;
    session.backend = backend.get();
    return true;
}
```

## Centralized Formatting (frontend.h)

Color and indentation are centralized so CLI and TUI are consistent:

```cpp
/// Logical colors for frontend output
enum class FrontendColor {
    DEFAULT,  // White/default terminal color
    GREEN,    // User input, tool results
    YELLOW,   // Tool calls
    RED,      // Errors, system warnings
    CYAN,     // Code blocks
    GRAY      // Thinking, dim text
};

/// Get color for a callback event type
FrontendColor get_color_for_event(CallbackEvent event);

/// Get indentation (spaces) for a callback event type
int get_indent_for_event(CallbackEvent event);
```

### Color Mapping

| CallbackEvent | FrontendColor |
|--------------|---------------|
| USER_PROMPT | GREEN |
| TOOL_CALL | YELLOW |
| TOOL_RESULT | GREEN |
| ERROR | RED |
| SYSTEM | RED |
| THINKING | GRAY |
| CODEBLOCK | CYAN |
| CONTENT | DEFAULT |

### Indentation

| CallbackEvent | Spaces |
|--------------|--------|
| USER_PROMPT, SYSTEM, ERROR | 0 |
| TOOL_CALL, CONTENT, THINKING, STATS | 2 |
| TOOL_RESULT, CODEBLOCK | 4 |

CLI maps FrontendColor to ANSI codes; TUI maps to ncurses color pairs.

## CLI Frontend

Interactive terminal mode with tool execution.

```cpp
class CLI : public Frontend {
public:
    void init(bool no_mcp, bool no_tools) override {
        Frontend::init_tools(session, tools, no_mcp, no_tools);
    }

    int run(Provider* cmdline_provider = nullptr) override;

    // Tool management
    Tools tools;

    // Output helpers
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& result);
    void show_error(const std::string& error);

    // Slash commands
    bool handle_slash_commands(const std::string& input);
};
```

### CLI Callback Setup

```cpp
int CLI::run(Provider* cmdline_provider) {
    // Set up callback before connecting
    callback = [this](CallbackEvent event, const std::string& content,
                      const std::string& name, const std::string& id) -> bool {
        switch (event) {
            case CallbackEvent::CONTENT:
            case CallbackEvent::CODEBLOCK:
            case CallbackEvent::THINKING:
                write_colored(content, get_color_for_event(event));
                break;
            case CallbackEvent::TOOL_CALL:
                show_tool_call(name, content);
                break;
            case CallbackEvent::ERROR:
                show_error(content);
                break;
            case CallbackEvent::STOP:
                // Generation complete
                break;
            // ... handle other events
        }
        return !g_generation_cancelled;
    };

    // Connect to provider (uses callback)
    connect_provider(provider_name);

    // Main loop
    while (true) {
        std::string input = read_input();
        if (handle_slash_commands(input)) continue;

        session.add_message(Message::USER, input);
        // ... handle tool calls
    }
}
```

### CLI Run Loop

```cpp
int CLI::run(Provider* cmdline_provider) {
    while (true) {
        // 1. Get user input
        std::string input = tio.read(">");

        // 2. Handle slash commands
        if (handle_slash_commands(input)) continue;

        // 3. Add message (triggers generation via backend)
        session.add_message(Message::USER, input);

        // 4. Handle tool calls if any
        while (!backend->pending_tool_calls.empty()) {
            auto& tc = backend->pending_tool_calls.front();
            ToolResult result = execute_tool(tools, tc.name, tc.parameters, tc.tool_call_id);
            session.add_message(Message::TOOL_RESPONSE, result.output, tc.name, tc.tool_call_id);
            backend->pending_tool_calls.erase(backend->pending_tool_calls.begin());
        }
    }
}
```

## Files

- `frontend.h` / `frontend.cpp` - Base class and init_tools()
- `frontends/cli.h` / `frontends/cli.cpp` - CLI implementation
- `server.h` / `server.cpp` - Server base class
- `frontends/api_server.h` / `frontends/api_server.cpp` - API server
- `frontends/cli_server.h` / `frontends/cli_server.cpp` - CLI server

## Version History

- **2.13.0** - Added init_tools() to eliminate CLI/CLIServer duplication
- **2.6.0** - Added Frontend abstraction and factory pattern
- **2.5.0** - Original CLI-only implementation
