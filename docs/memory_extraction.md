# Memory Extraction Module

## Overview

Background thread that automatically extracts useful facts from conversations and stores them in the RAG database. Complements the RAG context injection (read side, v2.27.0) with an automated write side -- ensuring ALL conversations get processed for useful information, not just when the model explicitly calls `store_memory`.

Added in v2.28.0.

## Architecture

### Components

- **MemoryExtractionThread** (`memory_extraction.h/cpp`): Worker thread that processes conversation segments via an external LLM API to extract Q/A fact pairs, then stores them using `RAGManager::store_memory()`.
- **ExtractionWorkItem**: Struct containing a deep copy of session messages, user_id, session_id, timestamp, and start_index for incremental processing.

### Threading Model

Single background worker thread consuming from a `ThreadQueue<ExtractionWorkItem>`:
- Worker calls `wait_for_and_pop(1s)` -- timeout allows checking the `running` flag for clean shutdown.
- Each work item is self-contained (deep-copied messages) -- no shared references to session state.
- Thread-local `RAGManager::current_user_id` is set per work item for multi-tenant isolation.
- SQLite uses WAL mode + 5s busy timeout for concurrent reader (main thread) + writer (extraction thread).

### Extraction Scope by Frontend

| Frontend    | Session Type | Scope per Work Item | Trigger |
|-------------|-------------|---------------------|---------|
| API Server  | Ephemeral   | Last user+assistant pair (2 msgs) | After each response |
| CLI / TUI   | Persistent  | Incremental since last extraction | Turn count threshold / idle timeout |
| CLI Server  | Persistent  | Incremental since last extraction | After each response |

### Queue Limit

Configurable via `memory_extraction_queue_limit`:
- `0` (default): Unlimited queue depth
- `>0`: When queue reaches limit, oldest item is dropped before new one is enqueued

### Extraction API Call

Non-streaming POST to `{memory_extraction_endpoint}/v1/chat/completions` with:
- System prompt requesting Q/A fact extraction
- Conversation text formatted as `User: ...\nAssistant: ...\n`
- Tool/function/system messages are filtered out
- `[context: ...]` postfix stripped from user messages

### Response Parsing

Lines starting with `Q: ` begin a question, lines starting with `A: ` provide the answer. Pairs are stored via `RAGManager::store_memory()`. Response of "NONE" means no extractable facts -- skipped.

## Multi-Tenant Isolation (user_id)

Thread-local `RAGManager::current_user_id` controls which user's memory space is accessed:
- **CLI/TUI/CLI-Server**: `config.user_id` if set, otherwise auto-detected as `hostname:username`
- **API server**: OpenAI `user` field from request > API key `name` > `config.user_id` > `"unknown"`
- **Extraction thread**: Set from work item's `user_id` before processing

When user_id is `"unknown"`, all RAG operations are skipped (no storage, no search, no injection, no extraction).

To share the same memory pool across platforms, set `user_id` to a common value in config on each machine:
```
shepherd config set user_id steve
```

All RAG operations (archive_turn, search, clear_memory) filter by user_id in both SQLite and PostgreSQL backends.

Schema change: `conversations` table has new `user_id TEXT NOT NULL DEFAULT 'local'` column with index. Existing databases are migrated automatically (idempotent ALTER TABLE).

## Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `user_id` | string | `""` | Common identity across platforms (empty = auto-detect) |
| `memory_extraction` | bool | `false` | Enable/disable extraction thread |
| `memory_extraction_model` | string | `""` | Model name for extraction API |
| `memory_extraction_endpoint` | string | `""` | OpenAI-compatible API base URL |
| `memory_extraction_api_key` | string | `""` | Bearer token (empty = no auth) |
| `memory_extraction_max_tokens` | int | `512` | Max tokens for extraction response |
| `memory_extraction_temperature` | double | `0.1` | Low temp for factual extraction |
| `memory_extraction_min_turns` | int | `2` | Min new user turns before triggering |
| `memory_extraction_idle_timeout` | int | `180` | Seconds idle before CLI/TUI trigger (0=disabled) |
| `memory_extraction_max_turns` | int | `20` | Max conversation turns per extraction call |
| `memory_extraction_queue_limit` | int | `0` | Max queued items (0=unlimited, >0=drop oldest) |

Thread will not start if `memory_extraction` is true but `memory_extraction_endpoint` or `memory_extraction_model` is empty.

## Error Handling

- All errors logged via `dout(1)`, never crash
- API unreachable: log, drop item, continue
- Invalid response: log, discard
- "NONE" response: normal, skip
- Entire worker loop body wrapped in try/catch

## Files

| File | Role |
|------|------|
| `memory_extraction.h` | ExtractionWorkItem struct + MemoryExtractionThread class |
| `memory_extraction.cpp` | Worker loop, API call, response parsing, storage |
| `rag.h` / `rag.cpp` | Thread-local user_id on RAGManager |
| `rag/sqlite_backend.cpp` | WAL mode, user_id column, filtered SQL queries |
| `rag/postgresql_backend.cpp` | user_id column, filtered prepared statements |
| `config.h` / `config.cpp` | 10 new config fields |
| `frontend.h` / `frontend.cpp` | Thread lifecycle, queue methods |
| `frontends/cli.cpp` | Queue after response, idle timeout, flush on exit |
| `generation_thread.cpp` | Queue after TUI generation |
| `frontends/api_server.cpp` | Set user_id, queue last exchange |
| `frontends/cli_server.cpp` | Queue after generation |
