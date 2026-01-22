# Configuration Refactor: Unified Config with MSI and SMCP Support

## Overview

Refactored Shepherd's configuration system to support:
1. **Unified config format** - All configuration in a single JSON file
2. **Azure Managed Identity (MSI)** - Load config from Azure Key Vault
3. **SMCP protocol** - Secure credential injection for MCP servers

## CLI Usage

```bash
# Local file (default)
shepherd

# Azure Key Vault via Managed Identity
shepherd --config msi --kv <vault-name>
```

## Unified Config Format

All configuration consolidated into one JSON structure:

```json
{
    "providers": [
        {
            "name": "azure-openai",
            "type": "openai",
            "model": "gpt-4o",
            "base_url": "https://...",
            "client_id": "...",
            "client_secret": "...",
            "token_url": "...",
            "token_scope": "...",
            "priority": 10
        }
    ],
    "mcp_servers": [
        {
            "name": "filesystem",
            "command": "mcp-filesystem",
            "args": ["/data"],
            "env": {"READONLY": "true"}
        }
    ],
    "smcp_servers": [
        {
            "name": "database",
            "command": "smcp-server-postgres",
            "credentials": {
                "DATABASE_URL": "postgresql://user:pass@host:5432/db"
            }
        }
    ],
    "streaming": true,
    "thinking": false,
    ...
}
```

### Key Sections

| Section | Description |
|---------|-------------|
| `providers[]` | LLM provider configurations (replaces separate provider files) |
| `mcp_servers[]` | Standard MCP servers with env vars |
| `smcp_servers[]` | MCP servers with SMCP credential injection |
| Settings | All other config options (streaming, thinking, etc.) |

## SMCP Protocol (v0.2)

Secure credential injection for MCP servers without exposing secrets in env vars or CLI args.

### Handshake

```
Parent (Shepherd)                    Child (SMCP Server)
      │                                    │
      │──────── fork + exec ──────────────►│
      │                                    │
      │◄───────── +READY ──────────────────┤
      │                                    │
      ├─ {"DATABASE_URL":"..."} ──────────►│
      │                                    │
      │◄───────── +OK ─────────────────────┤
      │                                    │
      │  ══════ MCP JSON-RPC ══════════    │
```

### Config Format

```json
{
    "smcp_servers": [
        {
            "name": "acmdev",
            "command": "smcp-server-postgres",
            "credentials": {
                "DATABASE_URL": "postgresql://...",
                "LOG_LEVEL": "INFO"
            }
        }
    ]
}
```

## Operating Modes

### Local File Mode (default)
- Config at `~/.config/shepherd/config.json`
- Slash commands (`/provider add`, `/mcp add`, `/config set`) can modify config
- Full read/write access

### Key Vault Mode (`--config msi --kv <vault>`)
- Config loaded from Azure Key Vault secret `shepherd-config`
- **Read-only** - slash commands that modify config are disabled
- For deployment/production use

## Files Created

| File | Purpose |
|------|---------|
| `azure_msi.h` | MSI token acquisition and Key Vault client declarations |
| `azure_msi.cpp` | IMDS token acquisition, Key Vault secret retrieval |

## Files Modified

| File | Changes |
|------|---------|
| `config.h` | Added `providers_json`, `smcp_config`, `SourceMode`, `is_read_only()` |
| `config.cpp` | Parse/save `providers[]` and `smcp_servers[]`, read-only check in `handle_config_args` |
| `main.cpp` | Added `--kv` argument, MSI config loading branch, set `source_mode` |
| `provider.cpp` | Load from `config->providers_json`, fallback to legacy files, read-only check |
| `mcp/mcp_config.h` | Added `SMCPServerEntry` struct |
| `mcp/mcp_config.cpp` | Parse/serialize `SMCPServerEntry`, read-only check in `handle_mcp_args` |
| `mcp/mcp_server.h` | Added `smcp_credentials` to Config, `perform_smcp_handshake()` method |
| `mcp/mcp_server.cpp` | Implemented SMCP handshake with timeout |
| `mcp/mcp.h` | Added `smcp_credentials` to ServerConfig |
| `mcp/mcp.cpp` | Load and connect SMCP servers from `config->smcp_config` |
| `CMakeLists.txt` | Added `azure_msi.cpp` to sources |

## Migration from Legacy Config

### Before (separate files)
```
~/.config/shepherd/
├── config.json           # Settings + MCP servers
├── providers/
│   ├── aztest.json       # Provider 1
│   └── localhost.json    # Provider 2
```

### After (unified)
```
~/.config/shepherd/
├── config.json           # Everything in one file
├── config.json.bak       # Backup of old config
├── providers/            # Legacy (fallback only)
```

## Azure Key Vault Setup

1. Create Key Vault secret named `shepherd-config`
2. Set secret value to the unified JSON config
3. Grant VM's managed identity "Key Vault Secrets User" role
4. Run: `shepherd --config msi --kv <vault-name>`

## Security Benefits

1. **No credentials in env vars** - Not visible in `/proc/<pid>/environ`
2. **No credentials in CLI args** - Not visible in `ps aux`
3. **No credentials on disk** (Key Vault mode) - Config fetched at startup
4. **SMCP handshake** - Credentials passed via stdin, exist only in memory
5. **Read-only mode** - Prevents accidental modification in production

## Dependencies

- **libcurl** - HTTP requests (already in project)
- **nlohmann/json** - JSON parsing (already in project)
- **smcp lib** - SMCP protocol support for MCP servers (`~/src/smcp/lib`)
- **smcp-server-postgres** - PostgreSQL MCP server with SMCP (`~/src/smcp/postgres`)

## Testing

```bash
# Build
make

# Test MSI error handling
./build/shepherd --config msi
# Error: --config msi requires --kv <vault-name>

# Test with Key Vault (on Azure VM)
./build/shepherd --config msi --kv my-vault

# Test local unified config
./build/shepherd
```
