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
- `conversations` - Stores conversation turns with tsvector search
- `facts` - Key-value fact storage

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

1. **Archiving**: After each conversation turn, the user message and assistant response are stored with a content hash to prevent duplicates.

2. **Search**: When the model uses the `search_memory` tool, the query is converted to a full-text search and the most relevant past conversations are returned.

3. **Facts**: The model can store and retrieve key-value facts using `set_fact`, `get_fact`, and `clear_fact` tools.

## Backend Detection

The backend is automatically selected based on the connection string:

| Pattern | Backend |
|---------|---------|
| Empty string | SQLite (default path) |
| File path | SQLite |
| `postgresql://...` | PostgreSQL |
| `postgres://...` | PostgreSQL |
