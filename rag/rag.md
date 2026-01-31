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

## Database Schema

### SQLite

```sql
-- Main storage table
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    content_hash TEXT UNIQUE NOT NULL
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE conversations_fts USING fts5(
    user_message,
    assistant_response,
    content='conversations',
    content_rowid='id'
);

-- Key-value fact storage
CREATE TABLE facts (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

### PostgreSQL

```sql
CREATE TABLE conversations (
    id BIGSERIAL PRIMARY KEY,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    content_hash CHAR(64) UNIQUE NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', user_message), 'A') ||
        setweight(to_tsvector('english', assistant_response), 'B')
    ) STORED
);

CREATE INDEX conversations_search_idx ON conversations USING GIN(search_vector);
CREATE INDEX conversations_timestamp_idx ON conversations(timestamp);

CREATE TABLE facts (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);
```

## Search Implementation

### SQLite (FTS5)
- Uses BM25 ranking algorithm
- Stop words filtered before query
- Keywords joined with OR for broad matching
- Score normalized to 0.0-1.0 range

### PostgreSQL (tsvector/tsquery)
- Uses ts_rank with normalization
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

// Archive a conversation turn
void RAGManager::archive_turn(const ConversationTurn& turn);

// Search for relevant context
std::vector<SearchResult> RAGManager::search(const std::string& query, int max_results);

// Fact storage
void RAGManager::set_fact(const std::string& key, const std::string& value);
std::string RAGManager::get_fact(const std::string& key);
bool RAGManager::clear_fact(const std::string& key);
```

## History

- **v2.23.0**: Added PostgreSQL backend support with abstract DatabaseBackend interface
- **v2.22.x**: Initial SQLite-only implementation with FTS5
