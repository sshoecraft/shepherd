# RAG (Retrieval-Augmented Generation) Module

## Overview

The RAG module provides conversation memory and fact storage for Shepherd. It archives conversation turns and enables semantic search to retrieve relevant context from past interactions.

## Architecture

### Database Backend Abstraction

The module uses a polymorphic backend architecture to support multiple database systems:

```
DatabaseBackend (abstract interface)
    ├── SQLiteBackend (default)
    └── PostgreSQLBackend (optional, compile-time)
```

### Files

| File | Purpose |
|------|---------|
| `rag.h` | Public API: RAGManager, SearchResult, ConversationTurn |
| `rag.cpp` | RAGManager implementation, tool handlers |
| `rag/database_backend.h` | Abstract DatabaseBackend interface |
| `rag/sqlite_backend.h/.cpp` | SQLite implementation with FTS5 |
| `rag/postgresql_backend.h/.cpp` | PostgreSQL implementation with tsvector |
| `rag/database_factory.cpp` | Factory for backend creation |

### Backend Selection

The factory auto-detects backend type from connection string:

- **SQLite** (default): File paths or empty string
  - `""` - uses default XDG path (`~/.local/share/shepherd/memory.db`)
  - `/path/to/database.db` - explicit file path

- **PostgreSQL**: Connection URLs (requires `-DENABLE_POSTGRESQL=ON`)
  - `postgresql://user:pass@host:5432/dbname`
  - `postgres://user:pass@host:5432/dbname`
  - Optional `schema` parameter to set search_path: `postgresql://...?schema=myschema`

## Database Schema

### SQLite

```sql
-- Context storage table (Q/A pairs from extraction)
CREATE TABLE context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    content_hash TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'local'
);

-- FTS5 virtual table for full-text search with time-based relevancy
CREATE VIRTUAL TABLE context_fts USING fts5(
    user_message,
    assistant_response,
    content='context',
    content_rowid='id'
);

-- Key-value fact storage (user-scoped, durable identity facts)
CREATE TABLE facts (
    user_id TEXT NOT NULL DEFAULT 'local',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, key)
);
```

### PostgreSQL

```sql
CREATE TABLE context (
    id BIGSERIAL PRIMARY KEY,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    content_hash CHAR(64) UNIQUE NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'local',
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', user_message), 'A') ||
        setweight(to_tsvector('english', assistant_response), 'B')
    ) STORED
);

CREATE INDEX context_search_idx ON context USING GIN(search_vector);
CREATE INDEX context_timestamp_idx ON context(timestamp);
CREATE INDEX context_user_id_idx ON context(user_id);

CREATE TABLE facts (
    user_id TEXT NOT NULL DEFAULT 'local',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    PRIMARY KEY (user_id, key)
);
```

## Search Implementation

### SQLite (FTS5)
- Uses BM25 ranking algorithm combined with time-based recency
- Recency factor: `1.0 + age_days/30.0` -- older entries get penalized
- Stop words filtered before query
- Keywords joined with OR for broad matching
- Score normalized to 0.0-1.0 range

### PostgreSQL (tsvector/tsquery)
- Uses ts_rank_cd combined with time-based recency
- Recency factor: `1.0 / (1.0 + age_days/30.0)` -- older entries get penalized
- Weighted search: user messages (A), assistant responses (B)
- plainto_tsquery for natural language queries
- GIN index for fast lookups

## Configuration

In `~/.config/shepherd/config.json`:

```json
{
    "memory_database": "",
    "max_db_size": "10G"
}
```

- `memory_database`: Connection string (empty = SQLite default)
- `max_db_size`: Maximum database size before pruning (SQLite only)

## Building

### SQLite Only (Default)
```bash
cmake ..
make
```

### With PostgreSQL Support
```bash
# Install dependencies
sudo apt install libpq-dev  # Debian/Ubuntu
sudo dnf install postgresql-devel  # Fedora/RHEL

# Build
cmake -DENABLE_POSTGRESQL=ON ..
make
```

## API

### RAGManager (Static Interface)

```cpp
// Initialize with connection string
bool RAGManager::initialize(const std::string& db_path, size_t max_db_size);

// Shutdown and cleanup
void RAGManager::shutdown();

// Archive a conversation turn (user_id for multi-tenant isolation)
void RAGManager::archive_turn(const ConversationTurn& turn, const std::string& user_id);

// Search for relevant context
std::vector<SearchResult> RAGManager::search_memory(const std::string& query, int max_results, const std::string& user_id);

// Memory storage
void RAGManager::store_memory(const std::string& question, const std::string& answer, const std::string& user_id);
bool RAGManager::clear_memory(const std::string& question, const std::string& user_id);

// Fact storage (user-scoped)
void RAGManager::set_fact(const std::string& key, const std::string& value, const std::string& user_id);
std::string RAGManager::get_fact(const std::string& key, const std::string& user_id);
bool RAGManager::clear_fact(const std::string& key, const std::string& user_id);
```

## Multi-Tenant Isolation

All write/search/clear operations filter by `user_id`, passed explicitly as a parameter:
- **CLI/TUI/CLI-Server**: `session.user_id` passed through `execute_tool()` and `enrich_with_rag_context()`
- **API server**: `request_session.user_id` set from API key name / request `user` field
- **Memory extraction thread**: `item->user_id` passed directly to `store_memory()`
- **Tool execution**: `user_id` injected as `_user_id` into tool args via `Tools::execute()`

See `memory_extraction.md` for the background extraction system.

## History

- **v2.32.0**: Renamed `conversations` table to `context` (idempotent migration). Added time-based recency to search scoring. Added `get_all_facts()` to DatabaseBackend interface. Disabled `archive_turn` during eviction (extraction thread handles memory capture). Extraction now produces structured JSON with separate facts and context. Injection presents `[facts: ...]` and `[context: ...]` separately.
- **v2.30.1**: Added user_id to facts table for multi-tenant isolation (composite PK on user_id, key)
- **v2.30.0**: Replaced thread_local user_id with explicit parameter passing through entire call chain
- **v2.28.0**: Added user_id column for multi-tenant isolation, WAL mode for SQLite concurrent access, idempotent schema migration
- **v2.25.1**: Added `schema` parameter to PostgreSQL connection string for search_path
- **v2.23.0**: Added PostgreSQL backend support with abstract DatabaseBackend interface
- **v2.22.x**: Initial SQLite-only implementation with FTS5
