# Command Consolidation Pattern

## Overview

Shepherd uses a command consolidation pattern where each command module (provider, model, config, scheduler, MCP) has a `handle_X_args()` function. Both CLI subcommands (`shepherd provider list`) and interactive slash commands (`/provider list`) call the same implementation, eliminating duplicate code.

## Pattern

### Declaration

Each command module declares its handler in the header file:

```cpp
// provider.h
int handle_provider_args(const std::vector<std::string>& args,
                         std::unique_ptr<Backend>* backend = nullptr,
                         Session* session = nullptr);

int handle_model_args(const std::vector<std::string>& args,
                      std::unique_ptr<Backend>* backend = nullptr);

// config.h
int handle_config_args(const std::vector<std::string>& args);

// scheduler.h
int handle_sched_args(const std::vector<std::string>& args);

// mcp/mcp_config.h
int handle_mcp_args(const std::vector<std::string>& args);
```

### CLI Subcommand Usage (main.cpp)

```cpp
// shepherd provider list
static int handle_provider_subcommand(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 2; i < argc; i++) {
        args.push_back(argv[i]);
    }
    return handle_provider_args(args);
}
```

### Interactive Slash Command Usage (cli.cpp)

```cpp
// /provider list
bool CLI::handle_slash_commands(const std::string& input,
                                std::unique_ptr<Backend>& backend,
                                Session& session) {
    // Parse command and args...

    if (cmd == "/provider") {
        handle_provider_args(args, &backend, &session);
        return true;
    }
    // ...
}
```

## Commands

### Provider Commands

| CLI | Interactive | Handler |
|-----|-------------|---------|
| `shepherd provider list` | `/provider list` | `handle_provider_args()` |
| `shepherd provider add <type> ...` | `/provider add <type> ...` | `handle_provider_args()` |
| `shepherd provider show [name]` | `/provider show [name]` | `handle_provider_args()` |
| `shepherd provider use <name>` | `/provider use <name>` | `handle_provider_args()` |
| `shepherd provider next` | `/provider next` | `handle_provider_args()` |

### Model Commands

| CLI | Interactive | Handler |
|-----|-------------|---------|
| `shepherd model` | `/model` | `handle_model_args()` |
| `shepherd model list` | `/model list` | `handle_model_args()` |
| `shepherd model set <name>` | `/model set <name>` | `handle_model_args()` |

### Config Commands

| CLI | Interactive | Handler |
|-----|-------------|---------|
| `shepherd config` | `/config` | `handle_config_args()` |
| `shepherd config show` | `/config show` | `handle_config_args()` |
| `shepherd config set <key> <value>` | `/config set <key> <value>` | `handle_config_args()` |

### Scheduler Commands

| CLI | Interactive | Handler |
|-----|-------------|---------|
| `shepherd sched` | `/sched` | `handle_sched_args()` |
| `shepherd sched list` | `/sched list` | `handle_sched_args()` |
| `shepherd sched add <name> "<cron>" "<prompt>"` | `/sched add ...` | `handle_sched_args()` |
| `shepherd sched remove <name>` | `/sched remove <name>` | `handle_sched_args()` |
| `shepherd sched enable <name>` | `/sched enable <name>` | `handle_sched_args()` |
| `shepherd sched disable <name>` | `/sched disable <name>` | `handle_sched_args()` |

### MCP Commands

| CLI | Interactive | Handler |
|-----|-------------|---------|
| `shepherd mcp` | N/A | `handle_mcp_args()` |
| `shepherd mcp list` | N/A | `handle_mcp_args()` |
| `shepherd mcp add <name> <cmd> [args]` | N/A | `handle_mcp_args()` |
| `shepherd mcp remove <name>` | N/A | `handle_mcp_args()` |
| `shepherd mcp test <name>` | N/A | `handle_mcp_args()` |

## Benefits

1. **No Code Duplication** - Single implementation serves both CLI and interactive modes
2. **Consistent Behavior** - Same logic ensures identical results
3. **Easy Maintenance** - Bug fixes apply to both interfaces
4. **Testability** - Handler functions can be unit tested directly

## Files

| Module | Header | Implementation |
|--------|--------|----------------|
| Provider | `provider.h` | `provider.cpp` |
| Config | `config.h` | `config.cpp` |
| Scheduler | `scheduler.h` | `scheduler.cpp` |
| MCP | `mcp/mcp_config.h` | `mcp/mcp_config.cpp` |
| CLI Slash | `cli.h` | `cli.cpp` |

## Version History

- **2.6.0** - Command consolidation pattern implemented
