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

    // Unified message and generation handling (for CLI, TUI, CLIServer - NOT APIServer)
    void add_message_to_session(Message::Role role, const std::string& content,
                                const std::string& tool_name = "",
                                const std::string& tool_call_id = "");
    bool generate_response();

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

## Unified Generation Flow

All stateful frontends (CLI, TUI, CLIServer) use the same generation path:

```cpp
// 1. Frontend adds user message to session
add_message_to_session(Message::USER, input);

// 2. Frontend calls generate_response()
generate_response();
```

### generate_response() internals

The `Frontend::generate_response()` method handles:

1. **Proactive eviction** - Checks if context is nearly full before generating
2. **Callback wrapping** - Wraps the frontend's callback to:
   - Accumulate content from CONTENT/THINKING/CODEBLOCK events
   - Add assistant message to session on STOP (before TOOL_CALL callbacks fire)
3. **Backend call** - Calls `backend->generate_from_session(session, max_tokens)`
4. **Reactive eviction** - Handles ContextFullException by evicting and retrying

The assistant message is added in the STOP callback (not after generate_from_session returns) because:
- TOOL_CALL callbacks fire after STOP but before generate_from_session returns
- TOOL_CALL handlers may call generate_response() recursively
- This recursive call would clear backend state before the original call can use it

### Tool Call Flow

When the model requests a tool call:

```
generate_response() called
    backend->generate_from_session()
        ... generates response with tool call ...
        record_tool_call()                    # Backend records tool call
        callback(STOP)                        # Wrapper adds assistant message HERE
        callback(TOOL_CALL)                   # Frontend handles tool execution
            execute_tool()
            add_message_to_session(TOOL_RESPONSE)
            generate_response()               # Recursive call for follow-up
                ... new generation ...
    generate_from_session() returns
generate_response() returns
```

## Files

- `frontend.h` / `frontend.cpp` - Base class and init_tools()
- `frontends/cli.h` / `frontends/cli.cpp` - CLI implementation
- `server.h` / `server.cpp` - Server base class
- `frontends/api_server.h` / `frontends/api_server.cpp` - API server
- `frontends/cli_server.h` / `frontends/cli_server.cpp` - CLI server

## Version History

- **2.22.2** - Skip local tool init for CLI backend (tools provided by remote server)
- **2.14.0** - Added unified `add_message_to_session()` and `generate_response()` methods; assistant messages now added in STOP callback to handle recursive tool call scenarios
- **2.13.0** - Added init_tools() to eliminate CLI/CLIServer duplication
- **2.6.0** - Added Frontend abstraction and factory pattern
- **2.5.0** - Original CLI-only implementation
