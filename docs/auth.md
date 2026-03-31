# API Key Authentication Module

## Overview

The auth module provides optional API key authentication for Shepherd's server modes (APIServer and CLIServer). When keys are configured, requests require an `Authorization: Bearer <key>` header. Without keys, servers remain open (backward compatible).

## Architecture

### Key Store Abstraction

```
KeyStore (abstract)
├── NoneKeyStore     - No authentication (default, no --apikey-store)
├── JsonKeyStore     - Local JSON file (apikey-store file://)
├── PGKeyStore       - PostgreSQL database (apikey-store postgresql://)
└── MsiKeyStore      - Azure Key Vault (apikey-store msi://)
```

### Configuration

Authentication is configured via the `--apikey-store` flag using a URI scheme:

| URI | Backend |
|-----|---------|
| (omitted) | NoneKeyStore - no auth |
| `file://` | JsonKeyStore - default path `~/.config/shepherd/api_keys.json` |
| `file:///path/to/keys.json` | JsonKeyStore - custom file path |
| `postgresql://user:pass@host/db` | PGKeyStore - PostgreSQL table |
| `msi://vault-name` | MsiKeyStore - Azure Key Vault |

### Key Storage Format (JSON)

Keys are stored in `~/.config/shepherd/api_keys.json` (or custom path):

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

### Key Storage Format (PostgreSQL)

```sql
CREATE TABLE IF NOT EXISTS api_keys (
    key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    notes TEXT DEFAULT '',
    created TEXT NOT NULL,
    permissions JSONB DEFAULT '{}'
);
```

The API key itself is the map key / primary key for O(1) lookup. The `permissions` field is reserved for future authorization features.

### Key Format

- Prefix: `sk-` (OpenAI convention)
- Random: 32 characters from `[a-zA-Z0-9]`
- Source: `/dev/urandom`
- Total: 35 characters

## Files

| File | Purpose |
|------|---------|
| `auth.h` | KeyStore interface, NoneKeyStore, JsonKeyStore, MsiKeyStore, ApiKeyEntry |
| `auth.cpp` | Key generation, validation, JSON load/save, URI factory |
| `auth_pg.h` | PGKeyStore class declaration (ENABLE_POSTGRESQL gated) |
| `auth_pg.cpp` | PGKeyStore implementation using libpq |

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
# Start with JSON key authentication (default path)
shepherd --apiserver --apikey-store file://

# Start with JSON key authentication (custom path)
shepherd --apiserver --apikey-store file:///etc/shepherd/keys.json

# Start with PostgreSQL key store
shepherd --apiserver --apikey-store "postgresql://user:pass@localhost/shepherd"

# Start without auth (default)
shepherd --apiserver
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

1. Server initializes `KeyStore` from `--apikey-store` URI
2. Pre-routing handler intercepts all requests (if auth enabled)
3. Public endpoints (`/health`, `/v1/models`) bypass auth
4. Protected endpoints require valid `Authorization: Bearer <key>` header
5. Invalid/missing key returns 401 with OpenAI-compatible error JSON

### PGKeyStore

- Uses libpq directly (same pattern as `rag/postgresql_backend.cpp`)
- Connection via `PQconnectdb()` with prepared statements
- Lazy-loads all keys on first validation call (cached in memory)
- Only compiled when `ENABLE_POSTGRESQL` cmake option is on
- Attempts postgresql:// without ENABLE_POSTGRESQL gives a clear error

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
- **2.39.0**: Replaced `--auth-mode` with URI-based `--apikey-store`, added PGKeyStore
