# Backend Creation and Initialization Refactoring Plan

## Goal

Refactor provider/backend initialization to:
1. Have Frontend own the providers list
2. Provider class represents a single provider (not a collection)
3. Clean separation between command-line provider creation and disk-loaded providers

## Current Architecture Issues

1. `Provider` class is misnamed - it's actually a provider collection/registry
2. `ProviderConfig` is the actual single provider representation
3. Provider loading/management scattered between main.cpp and provider.cpp
4. `handle_provider_args()` creates a new Provider instance each time, loads from disk

## New Architecture

### Provider Class

`Provider` represents a single provider configuration. Merge current `ProviderConfig` hierarchy into `Provider`:

```cpp
class Provider {
public:
    std::string name;
    std::string type;           // llamacpp, openai, anthropic, etc.
    std::string model;
    int priority = 100;
    size_t context_size = 0;

    // Type-specific fields (union-like, based on type)
    // API providers
    std::string api_key;
    std::string base_url;
    float temperature = 0.7f;
    // ... etc

    // Local providers (llamacpp, tensorrt)
    std::string model_path;
    int gpu_layers = -1;
    int tp = 1;
    int pp = 1;
    // ... etc

    // Static factory to load all providers from disk
    static std::vector<Provider> load_providers();

    // Create backend from this provider
    std::unique_ptr<Backend> connect(Session& session);

    // Serialization
    nlohmann::json to_json() const;
    static Provider from_json(const nlohmann::json& j);
};
```

### Frontend Class

Frontend owns the providers list:

```cpp
class Frontend {
public:
    static std::unique_ptr<Frontend> create(
        const std::string& mode,
        const std::string& host,
        int port,
        Provider* cmdline_provider = nullptr  // Optional command-line provider
    );

    // ...

protected:
    std::vector<Provider> providers;  // Loaded from disk + optional cmdline
    std::string current_provider;     // Name of active provider
};
```

### Initialization Flow

**main.cpp:**
```cpp
// Parse command-line args into override struct (as now)

// If command-line specifies backend/model/etc, create ephemeral provider
Provider* cmdline_provider = nullptr;
Provider cmdline;
if (has_cmdline_override) {
    cmdline = Provider::from_config();  // Creates from global config state
    cmdline.name = "_cmdline";
    cmdline.priority = 0;
    cmdline_provider = &cmdline;
}

// Create frontend, passing optional cmdline provider
auto frontend = Frontend::create(mode, host, port, cmdline_provider);

// Frontend::create() internally:
//   1. Calls Provider::load_providers() to get disk providers
//   2. If cmdline_provider passed, adds it to the list
//   3. Stores in this->providers

frontend->init(no_mcp, no_tools);
frontend->run(session);
```

**Frontend::create():**
```cpp
std::unique_ptr<Frontend> Frontend::create(
    const std::string& mode,
    const std::string& host,
    int port,
    Provider* cmdline_provider)
{
    std::unique_ptr<Frontend> frontend;

    if (mode == "cli") {
        frontend = std::make_unique<CLI>();
    } else if (mode == "api-server") {
        frontend = std::make_unique<APIServer>(host, port);
    } else if (mode == "cli-server") {
        frontend = std::make_unique<CLIServer>(host, port);
    } else {
        throw std::runtime_error("Invalid frontend mode: " + mode);
    }

    // Load providers from disk
    frontend->providers = Provider::load_providers();

    // Add command-line provider if specified
    if (cmdline_provider) {
        frontend->providers.insert(frontend->providers.begin(), *cmdline_provider);
    }

    return frontend;
}
```

### Backend Creation

Move backend creation logic into `Provider::connect()`:

```cpp
std::unique_ptr<Backend> Provider::connect(Session& session) {
    // Current logic from BackendFactory::create_from_provider()
    // + backend->initialize(session)
    // Returns fully initialized backend
}
```

### handle_provider_args() Changes

Instead of creating its own Provider instance, it receives the providers vector:

```cpp
int handle_provider_args(
    const std::vector<std::string>& args,
    std::vector<Provider>& providers,      // The frontend's provider list
    std::unique_ptr<Backend>* backend = nullptr,
    Session* session = nullptr);
```

## Files to Modify

### provider.h / provider.cpp
- Merge ProviderConfig into Provider
- Remove the old Provider class (the collection one)
- Add `static std::vector<Provider> load_providers()`
- Add `std::unique_ptr<Backend> connect(Session& session)`
- Update `handle_provider_args()` signature

### frontend.h / frontend.cpp
- Add `std::vector<Provider> providers` member
- Update `create()` to accept optional `Provider*`
- Load providers in `create()`

### cli.cpp
- Pass `providers` to `handle_provider_args()`

### main.cpp
- Create cmdline Provider if needed
- Pass to `Frontend::create()`
- Remove provider loading logic (moved to Frontend)

### backends/factory.h / factory.cpp
- May be simplified or removed if Provider::connect() handles creation

## Frontend::init() Changes

Move this logic from main.cpp into Frontend::init():

1. **Tool registration** - currently done in main.cpp with dynamic_cast to CLI/CLIServer
2. **Provider connection** - selecting and connecting to provider
3. **Provider tool registration** - registering ask_<provider> tools

New signature:
```cpp
virtual void init(Session& session, bool no_mcp = false, bool no_tools = false,
                  const std::string& provider_name = "");
```

After init(), Frontend owns:
- The connected Backend
- The providers list
- Current provider name

Then `run()` no longer needs backend passed in:
```cpp
virtual int run(Session& session) = 0;  // Backend owned by Frontend
```

## Migration Steps

1. Add new Provider class (can coexist with old temporarily)
2. Add `Provider::load_providers()` static method
3. Add `providers` member to Frontend
4. Update `Frontend::create()` signature and implementation
5. Move tool init and provider connection from main.cpp to Frontend::init()
6. Update main.cpp to use new flow
7. Update `handle_provider_args()` to use passed-in providers
8. Remove old Provider class and ProviderConfig hierarchy
9. Clean up BackendFactory if no longer needed

## Future Refactors (Out of Scope)

- Diamond inheritance for CLIServer (inherits from both CLI and Server)
  - Would eliminate duplicate Tools member and tool registration code
