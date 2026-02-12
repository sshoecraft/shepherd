# Memory Database

Shepherd includes a memory system that archives conversation turns and enables semantic search to retrieve relevant context from past interactions.

## Backends

### SQLite (Default)

SQLite with FTS5 full-text search. No additional dependencies required.

```json
{
    "memory_database": ""
}
```

An empty string uses the default path: `~/.local/share/shepherd/memory.db`

You can also specify an explicit path:

```json
{
    "memory_database": "/path/to/memory.db"
}
```

### PostgreSQL (Optional)

PostgreSQL with tsvector/tsquery full-text search. Useful for multi-machine deployments or when you want a centralized memory store.

#### Prerequisites

Install libpq development headers:

```bash
# Debian/Ubuntu
sudo apt install libpq-dev

# Fedora/RHEL
sudo dnf install postgresql-devel

# Arch
sudo pacman -S postgresql-libs
```

#### Building

Build Shepherd with PostgreSQL support:

```bash
cmake -DENABLE_POSTGRESQL=ON ..
make
```

#### Configuration

Set the connection string in your config:

```json
{
    "memory_database": "postgresql://user:password@localhost:5432/shepherd"
}
```

Connection string formats:
- `postgresql://user:pass@host:5432/dbname`
- `postgres://user:pass@host:5432/dbname`

#### Database Setup

Shepherd creates the required tables automatically on first connection. The PostgreSQL user needs CREATE TABLE permissions.

Required tables (auto-created):
- `context` - Stores Q/A pairs from extraction with full-text search
- `facts` - User-scoped key-value fact storage (composite PK on `user_id`, `key`)

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `memory_database` | string | `""` | Connection string (empty = SQLite default) |
| `max_db_size` | string | `"10G"` | Maximum database size before pruning (SQLite only) |

## Command Line

Override the memory database path:

```bash
shepherd --memory-database /path/to/memory.db
shepherd --memory-database "postgresql://user:pass@host/db"
```

## How It Works

1. **Extraction**: The background extraction thread processes conversation segments via an external LLM to extract Q/A pairs and key-value facts, storing them in the `context` and `facts` tables respectively.

2. **Injection**: Before each inference call, `enrich_with_rag_context()` searches the `context` table using keywords from the user's message and retrieves all facts for the user. Matching results are postfixed onto the user message as `[facts: ...]` and `[context: ...]`.

3. **Facts**: Durable key-value facts (name, location, preferences) scoped by `user_id`. Written by the extraction thread via `set_fact()` (INSERT OR REPLACE). All fact operations filter by `user_id` for multi-tenant isolation. Memory tools (`search_memory`, `set_fact`, `get_fact`, `clear_fact`, `store_memory`, `clear_memory`) are not registered by default. Pass `--memtools` to enable them (intended for the extraction provider or testing).

## Backend Detection

The backend is automatically selected based on the connection string:

| Pattern | Backend |
|---------|---------|
| Empty string | SQLite (default path) |
| File path | SQLite |
| `postgresql://...` | PostgreSQL |
| `postgres://...` | PostgreSQL |
