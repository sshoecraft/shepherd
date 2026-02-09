# RAG Context Injection

## Overview

Automatic RAG-based context injection enriches user messages with relevant stored
knowledge before inference. Shepherd searches the RAG backend using keywords
extracted from the current user message and postfixes matching results onto the
message content. The model sees enriched context and responds naturally — no
explicit tool calls needed for retrieval.

Added in v2.27.0.

## Architecture

```
User message arrives
        |
        v
add_message_to_session(Message::USER, input)
        |
        v
enrich_with_rag_context(session)
  - Extract keywords from last user message
  - Query RAG backend (FTS5/Postgres full-text search)
  - Filter by relevance threshold
  - If results: postfix [context: ...] onto user message
  - Recount tokens, update session totals
  - If no results: leave message unchanged (no-op)
        |
        v
generate_response() / prefill_session() (unchanged)
```

## Configuration

Three config fields control the feature:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rag_context_injection` | bool | `false` | Enable/disable the feature |
| `rag_relevance_threshold` | double | `0.3` | Minimum relevance score (0.0-1.0) |
| `rag_max_results` | int | `3` | Maximum results to inject |

Set via CLI:
```
shepherd config set rag_context_injection true
shepherd config set rag_relevance_threshold 0.3
shepherd config set rag_max_results 3
```

Or in `config.json`:
```json
{
    "rag_context_injection": true,
    "rag_relevance_threshold": 0.3,
    "rag_max_results": 3
}
```

## Implementation

### Files Modified

| File | Change |
|------|--------|
| `config.h` | Added 3 config fields |
| `config.cpp` | Defaults, JSON load/save, get/set/list support |
| `frontend.h` | Declared `enrich_with_rag_context(Session&)` and `extract_keywords(string)` |
| `frontend.cpp` | Core implementation (~115 lines) |
| `frontends/cli.cpp:399` | Inserted call between add_message and generate |
| `generation_thread.cpp:87` | Inserted guarded call (USER role only) |
| `frontends/cli_server.cpp:287` | Inserted call between add_message and generate |
| `frontends/api_server.cpp:433` | Inserted call after message parsing, before params |

### Core Functions

Both methods live on the `Frontend` base class (`frontend.h` / `frontend.cpp`).

**`Frontend::extract_keywords(const std::string& message)`** (static)

Extracts search-worthy terms from a user message:
1. Tokenize by whitespace
2. Strip punctuation, lowercase each word
3. Skip words <= 2 characters
4. Remove ~70 stop words (articles, pronouns, prepositions, modals, conversational fillers)
5. Return space-separated keywords

The stop word list is broader than the one in `sqlite_backend.cpp` because it handles
casual conversational queries ("what did I tell you about my project?") where aggressive
filtering yields better search precision.

**`Frontend::enrich_with_rag_context(Session& sess)`**

Guard checks (returns immediately if any fail):
- `config->rag_context_injection` must be true
- `RAGManager::is_initialized()` must be true
- Session must have messages
- Last user message index must be valid
- Last message must be USER role

Processing:
1. Extract keywords from last user message
2. Call `RAGManager::search_memory(keywords, config->rag_max_results)`
3. Filter results by `config->rag_relevance_threshold`
4. Build pipe-separated context string from results
5. Postfix onto message: `content + "\n\n[context: " + context + "]"`
6. Log enriched message to stdout as `[prompt+context]`
7. Recount tokens via `backend->count_message_tokens()`
8. Update `sess.total_tokens` and `sess.last_user_message_tokens` with delta

### Insertion Points

All four frontends insert the call between `add_message_to_session()` and
`generate_response()`:

**CLI** (`frontends/cli.cpp`):
```cpp
auto lock = backend->acquire_lock();
add_message_to_session(Message::USER, user_input);
enrich_with_rag_context(session);
generate_response();
```

**TUI** (`generation_thread.cpp`):
```cpp
frontend->add_message_to_session(role, content, ...);
if (current_request.role == Message::USER) {
    frontend->enrich_with_rag_context(frontend->session);
}
frontend->generate_response(max_tokens);
```

**CLI Server** (`frontends/cli_server.cpp`):
```cpp
state.server->add_message_to_session(Message::USER, prompt);
state.server->enrich_with_rag_context(state.server->session);
state.server->generate_response(max_tokens);
```

**API Server** (`frontends/api_server.cpp`):
```cpp
// After message parsing loop, before parameter parsing
enrich_with_rag_context(request_session);
```

## KV Cache Impact

### CLI/TUI (persistent session)
Context persists in `session.messages` -> KV cache mirror matches exactly ->
zero cache overhead. Only the genuinely new message + context gets decoded.

### API Server (ephemeral session)
Each request is a fresh session. The client sends un-enriched messages. On the
next request, the KV cache diverges at the previously-enriched user message,
which is near the END of the token sequence. The prefix match covers everything
before it (system prompt + all earlier turns). Only the last 1-2 turns get
re-decoded — negligible cost on a 128k conversation.

### What NOT to do
- Never inject into the system prompt — it's tokenized first, so any change
  invalidates the entire cache from position 0
- Never inject as a separate message — breaks user/assistant alternation

## Relationship to LARS

This feature replaces the RETRIEVAL side of LARS. The model no longer needs to
call `search_memory` to find stored knowledge — Shepherd injects it automatically.

The STORAGE side remains model-driven. Only the model can judge whether a
successful interaction is worth remembering. When it stores via `store_memory`,
the data enters the RAG backend and becomes available for future automatic injection.

- Read path: Shepherd (infrastructure, microseconds)
- Write path: Model (judgment, via tool calls)

## Logging

When context is injected, the server log shows:
```
[prompt+context] What temperature is the ecobee thermostat?

[context: User: ecobee route\nAssistant: route:ecobee/get_temp ...]
```

Debug level 1 (`-d`) shows keyword extraction, search results, threshold
filtering, and token count changes.

When no context is injected (no keywords, no results, below threshold),
nothing is logged — no noise.
