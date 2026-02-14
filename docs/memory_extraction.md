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
- `user_id` from each work item is passed explicitly to `RAGManager::store_memory()` for multi-tenant isolation.
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
- System prompt requesting fact extraction from user statements only
- Conversation text formatted as `User: ...\n` (USER messages only, v2.33.2)
- Assistant messages are excluded to prevent extraction model from storing assistant-generated content as user facts
- Tool/function/system messages are filtered out
- `[facts: ...]` and `[context: ...]` postfix stripped from user messages

### Response Parsing

The extraction model returns JSON with two sections:
```json
{
    "facts": {"name": "Steve", "location": "Texas"},
    "context": [{"q": "What IDE does the user prefer?", "a": "VS Code with vim keybindings"}]
}
```

- **Facts**: Key-value pairs stored via `RAGManager::set_fact()` (INSERT OR REPLACE -- newer value always wins, no duplicates)
- **Context**: Q/A pairs stored via `RAGManager::store_memory()` into the `context` table

Empty response: `{"facts": {}, "context": []}` -- nothing stored. Markdown code blocks around JSON are stripped automatically.

## Multi-Tenant Isolation (user_id)

`user_id` is passed explicitly through the call chain to all RAG operations:
- **CLI/TUI/CLI-Server**: `session.user_id` (from `config.user_id` if set, otherwise auto-detected as `hostname:username`)
- **API server**: `request_session.user_id` set from OpenAI `user` field > API key `name` > `config.user_id` > `"unknown"`
- **Extraction thread**: `item->user_id` passed directly to `RAGManager::store_memory()`
- **Tool execution**: `user_id` injected as `_user_id` parameter via `Tools::execute()` and read by memory tools

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
| `memory_extraction_retry_interval` | int | `5` | Seconds between retry attempts when API is unreachable |

The extraction thread starts whenever `memory_extraction_endpoint` and `memory_extraction_model` are both configured. The thread itself always runs; the provider's `memory` flag controls whether work items get queued (see below).

## Provider Memory Flag

As of v2.29.0, the provider config has a `memory` boolean (default `false`). When a provider connects, it sets both `config->rag_context_injection` and `config->memory_extraction` from this flag. This controls whether RAG context injection and memory extraction queueing are active for that provider.

| Provider target | `memory` setting | Rationale |
|---|---|---|
| Local model (llamacpp) | `true` | Only Shepherd provides RAG |
| vLLM / raw inference server | `true` | No memory on the server side |
| Remote Shepherd server (cli) | `false` | Server handles its own RAG |
| Anthropic/OpenAI/Gemini | User's choice | Shepherd RAG is separate from provider memory |

Set via CLI:
```
shepherd provider my-provider set memory true
```

## Retry Behavior

When the extraction API endpoint is unreachable (connection refused, timeout, HTTP error), the worker retries at a fixed interval configured by `memory_extraction_retry_interval` (default 5 seconds). This handles the case where the extraction model service starts after Shepherd.

- Retries indefinitely until success or shutdown
- Sleeps in 1-second increments for responsive shutdown (< 2s exit latency)
- Interval is read from config on each retry, so runtime changes take effect immediately

## Error Handling

- All errors logged via `dout(1)`, never crash
- API unreachable: retry at configured interval
- Invalid response: log, discard
- "NONE" response: normal, skip
- Entire worker loop body wrapped in try/catch

## Files

| File | Role |
|------|------|
| `memory_extraction.h` | ExtractionWorkItem struct + MemoryExtractionThread class |
| `memory_extraction.cpp` | Worker loop, API call, response parsing, storage |
| `rag.h` / `rag.cpp` | RAGManager static API with explicit user_id params |
| `rag/sqlite_backend.cpp` | WAL mode, user_id column, filtered SQL queries |
| `rag/postgresql_backend.cpp` | user_id column, filtered prepared statements |
| `config.h` / `config.cpp` | 11 config fields (including retry_interval) |
| `frontend.h` / `frontend.cpp` | Thread lifecycle, queue methods |
| `frontends/cli.cpp` | Queue after response, idle timeout, flush on exit |
| `generation_thread.cpp` | Queue after TUI generation |
| `frontends/api_server.cpp` | Set user_id, queue last exchange |
| `frontends/cli_server.cpp` | Queue after generation |
| `provider.h` / `provider.cpp` | Provider `memory` flag sets config globals on connect |

## History

- **v2.32.0**: Reworked extraction to produce structured JSON with separate `facts` (key-value pairs → `set_fact`) and `context` (Q/A pairs → `store_memory`). Archive_turn during eviction disabled (extraction thread handles it). Table renamed `conversations` → `context`. Injection now presents `[facts: ...]` and `[context: ...]` separately. Time-based relevancy added to context search.
- **v2.33.2**: Extraction input now contains only USER messages (assistant messages excluded). Prevents extraction model from storing assistant-generated content as user facts. Extraction prompt strengthened to reject transient data, operational state, and discussed content.
- **v2.31.1**: Fixed extraction prompt to only extract facts stated by the user, not facts about the assistant itself (e.g., assistant self-identifying as "Shepherd" was being stored as a fact)
- **v2.30.0**: Replaced thread_local user_id with explicit parameter passing through entire call chain
- **v2.29.0**: Added retry with configurable interval (`memory_extraction_retry_interval`), provider `memory` flag controls injection + extraction, thread starts based on endpoint+model presence (not `memory_extraction` bool)
- **v2.28.0**: Initial implementation with background extraction thread, multi-tenant user_id isolation
