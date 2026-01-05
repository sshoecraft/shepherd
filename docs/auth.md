# API Key Authentication Module

## Overview

The auth module provides optional API key authentication for Shepherd's server modes (APIServer and CLIServer). When keys are configured, requests require an `Authorization: Bearer <key>` header. Without keys, servers remain open (backward compatible).

## Architecture

### Key Store Abstraction

```
KeyStore (abstract)
├── NoneKeyStore     - No authentication (default)
├── JsonKeyStore     - Local JSON file storage
├── SqliteKeyStore   - SQLite database (future)
├── VaultKeyStore    - HashiCorp Vault (future)
└── ManagedKeyStore  - Azure/AWS managed identity (future)
```

### Key Storage Format

Keys are stored in `~/.config/shepherd/api_keys.json`:

```json
{
    "sk-a7B3xK9pQ2mN5vL8rT1wY6cE4hJ0gF12": {
        "name": "primary",
        "notes": "Production server",
        "created": "2026-01-03T12:00:00Z",
        "permissions": {}
    }
}
```

The API key itself is the map key for O(1) lookup. The `permissions` field is reserved for future authorization features.

### Key Format

- Prefix: `sk-` (OpenAI convention)
- Random: 32 characters from `[a-zA-Z0-9]`
- Source: `/dev/urandom`
- Total: 35 characters

## Files

| File | Purpose |
|------|---------|
| `auth.h` | KeyStore interface, NoneKeyStore, JsonKeyStore, ApiKeyEntry |
| `auth.cpp` | Key generation, validation, JSON load/save |

## CLI Usage

```bash
# Generate key
shepherd keygen --name production
shepherd keygen --name ci --notes "GitHub Actions"

# List keys (masked)
shepherd keygen list

# Remove key
shepherd keygen remove production
```

## Server Usage

```bash
# Start with JSON key authentication
shepherd --cliserver --auth-mode json

# Start without auth (default)
shepherd --cliserver --auth-mode none
```

## Client Configuration

Provider config in `~/.config/shepherd/providers/cli-server.json`:

```json
{
    "name": "cli-server",
    "type": "cli",
    "base_url": "http://server:8000",
    "api_key": "sk-xxxxx"
}
```

## Security

1. **Constant-time comparison** - Prevents timing attacks on key validation
2. **0600 permissions** - Keys file readable only by owner
3. **Keys never logged** - Not even partially masked in logs
4. **HTTPS recommended** - For production deployments

## Implementation Notes

### Authentication Flow

1. Server initializes `KeyStore` from `--auth-mode` parameter
2. Pre-routing handler intercepts all requests (if auth enabled)
3. Public endpoints (`/health`, `/v1/models`) bypass auth
4. Protected endpoints require valid `Authorization: Bearer <key>` header
5. Invalid/missing key returns 401 with OpenAI-compatible error JSON

### Error Response Format

```json
{
    "error": {
        "message": "Invalid API key",
        "type": "authentication_error",
        "code": "401"
    }
}
```

## Version History

- **2.17.0**: Initial implementation with `none` and `json` auth modes
