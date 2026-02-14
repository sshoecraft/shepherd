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
  - Get all facts for user → format as [facts: key=value, ...]
  - Extract keywords from last user message
  - Query context table (FTS5/Postgres full-text search with recency)
  - Filter by relevance threshold
  - If results: postfix [context: ...] onto user message
  - Recount tokens, update session totals
  - If nothing to inject: leave message unchanged (no-op)
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
1. Get all facts for user via `RAGManager::get_all_facts(sess.user_id)` → format as `[facts: key=value, ...]`
2. Extract keywords from last user message
3. Call `RAGManager::search_memory(keywords, config->rag_max_results, sess.user_id)` (searches `context` table with time-based recency)
4. Filter results by `config->rag_relevance_threshold`
5. Build injection string: facts first, then pipe-separated context results
6. Postfix onto message: `content + "\n\n[facts: ...]\n[context: ...]"`
7. Log enriched message to stdout as `[prompt+context]`
8. Recount tokens via `backend->count_message_tokens()`
9. Update `sess.total_tokens` and `sess.last_user_message_tokens` with delta

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
[prompt+context] What is my name?

[facts: name=Steve, location=Texas]
[context: User: What IDE does the user prefer?\nAssistant: VS Code with vim keybindings]
```

Debug level 1 (`-d`) shows keyword extraction, search results, threshold
filtering, and token count changes.

When no context is injected (no keywords, no results, below threshold),
nothing is logged — no noise.

## Injection Stripping (v2.31.0)

RAG context is now stripped from previous user messages each turn. Only the
current user message carries injected content. This prevents context bloat in
long conversations — without stripping, 200 messages each carrying 500 tokens
of injected context wastes 100K tokens.

### How It Works

1. Before injecting, `pre_injection_content` is saved on the Message
2. After the API call, delta tracking corrects the message's `.tokens` to
   the actual value (from `prompt_tokens - last_prompt_tokens`)
3. Next turn, before enriching the new message:
   - Previous messages with `pre_injection_content` set are restored
   - Token count is recalculated using per-message correction:
     `per_message_cpt = content.length() / tokens` (derived from delta-corrected actual)
     `new_tokens = restored_content.length() / per_message_cpt`
   - `session.total_tokens` is decremented by the freed tokens

### Delta Tracking (v2.31.0)

All API backends now use `ApiBackend::update_session_tokens()` for post-generation
token accounting. This method:

- Computes `delta = prompt_tokens - session.last_prompt_tokens` to derive actual
  per-message token counts (replaces the EMA-only estimates)
- Corrects `Message.tokens` for the most recently added user message
- Refines the EMA `chars_per_token` ratio (alpha=0.2, clamped to [2.0, 5.0])
  for better future estimates
- Replaces the duplicated overwrite code that was copy-pasted across OpenAI,
  Anthropic, and Gemini streaming paths

## Provider Memory Flag (v2.29.0)

As of v2.29.0, the provider config has a `memory` boolean (default `false`) that
controls both RAG context injection and memory extraction. When a provider connects,
`Provider::connect()` sets `config->rag_context_injection` and `config->memory_extraction`
from the provider's `memory` field.

This means injection is only active when the current provider has `memory: true`.
The RAG database is only initialized when a provider with `memory: true` connects
(v2.33.2) — providers with `memory: false` never open the database. If you switch
from a `memory: false` provider to a `memory: true` provider, RAG is initialized
at that point (fail-fast on connection errors).

The `--nomemory` command-line flag (v2.34.0) overrides the provider's memory setting,
disabling both injection and extraction regardless of the provider config. For CLI
client backends, this also sends `"memory": false` in requests to the server.

The API server and CLI server also support per-request `"memory": false` in the
request JSON, allowing clients to opt out of memory for individual requests.

For providers connecting to a remote Shepherd server (type `cli`), set `memory: false`
(the default) since the server handles its own RAG. For local models or raw inference
endpoints (vLLM), set `memory: true` to enable local RAG.

```
shepherd provider my-local-model set memory true
```
