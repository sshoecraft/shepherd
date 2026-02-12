# Refactor Memory Extraction to Use Provider/Backend

## Context

Currently `memory_extraction.cpp` does raw HTTP POST to an OpenAI-compatible endpoint via `HttpClient`, asks the model to return structured JSON, then manually parses the response and routes facts to `set_fact()` and context to `store_memory()`. This bypasses the entire Provider/Backend system that every other model interaction in Shepherd uses.

The `ask_` tools (APIToolAdapter in `tools/api_tools.cpp`) show the right pattern: create a backend via `Provider::connect()`, use a real session, and optionally provide tools.

## Goal

Replace the raw HTTP extraction with a proper Provider/Backend. Two phases:

### Phase 1: Use Provider/Backend for the API call

Replace `HttpClient` with `Provider::connect()`. The extraction model still returns text (JSON or whatever format), we still parse it. But the plumbing uses the same infrastructure as everything else — proper backend, session, error handling, retry, auth.

This works with any model regardless of tool-calling support.

### Phase 2: Add tool calling (if model supports it)

Give the extraction session `set_fact` and `store_memory` tools. The model calls them directly via structured tool calls. No text parsing needed. The tool execution loop (same pattern as APIToolAdapter) handles everything.

This requires a model that supports tool calling well. GLM-4 9B may or may not — needs testing.

## Current Flow

```
conversation text → raw HttpClient POST → JSON response → parse JSON → route to set_fact/store_memory
```

## Phase 1 Flow

```
conversation text → Provider::connect() backend → generate_from_session → text response → parse → route
```

## Phase 2 Flow (future, if model supports tools)

```
conversation text → Provider::connect() backend → session with tools → tool execution loop
                                                                        ↓
                                                          model calls set_fact("name", "Steve")
                                                          model calls store_memory("What IDE?", "VS Code")
                                                          loop executes tools directly
```

## Phase 1 Changes

### 1. Create extraction Provider from config

Build a `Provider` struct from the existing extraction config fields:
- `memory_extraction_endpoint` → `provider.base_url`
- `memory_extraction_model` → `provider.model`
- `memory_extraction_api_key` → `provider.api_key`
- `memory_extraction_temperature` → `provider.temperature`
- `memory_extraction_max_tokens` → `provider.max_tokens`
- Type: `"openai"` (OpenAI-compatible endpoint)

### 2. Replace call_extraction_api with backend generate

- `ensure_connected()` — lazy-init backend via `Provider::connect()` (reuse across calls)
- Set system prompt on session
- Add conversation text as user message
- `backend->generate_from_session()` — non-streaming, accumulate response
- Parse response same as now (JSON with facts/context)
- Clear session messages after each extraction

### 3. Update MemoryExtractionThread members

- Remove `call_extraction_api()` method
- Add `Provider extraction_provider` member
- Add `std::unique_ptr<Backend> extraction_backend` member
- Add `Session extraction_session` member
- Add `bool connected` flag
- Add `ensure_connected()` method
- Keep `parse_and_store_facts()` as-is (still needed in Phase 1)
- Keep the worker thread, queue, retry logic

### 4. Threading consideration

The backend created for extraction is dedicated to the background thread — no sharing with the main backend. SQLite WAL mode + busy_timeout already handles concurrent writer (extraction) + reader (main thread).

Streaming must be disabled for the extraction backend (set `config->streaming = false` temporarily during connect, same as APIToolAdapter does).

## Phase 2 Changes (future session)

### 5. Add tool subset to extraction session

Only expose these tools to the extraction model:
- `set_fact(key, value)` — store durable facts
- `store_memory(question, answer)` — store context Q/A pairs

No filesystem tools, no HTTP tools, no command tools.

### 6. Add tool execution loop

Same pattern as APIToolAdapter: generate, check for tool calls, execute them, feed results back, repeat.

### 7. Remove parse_and_store_facts

No longer needed — the model calls tools directly.

### 8. Simplify extraction prompt

```
You are a memory extraction system. Analyze the conversation below and store useful information.

Use set_fact for durable identity facts about the user that will still be true months from now:
- name, location, job, preferences, project names, technical choices
- Use lowercase_snake_case keys
- Only store facts explicitly stated or confirmed by the user
- Do NOT store facts about the assistant itself

Use store_memory for useful conversational context as question/answer pairs:
- Decisions made, procedures discussed, problems solved
- Each entry should be self-contained

Do NOT store:
- Transient data (readings, prices, timestamps)
- Tool call results or API responses
- Small talk, greetings, filler
- Information that was corrected or superseded

If there is nothing worth storing, simply respond that there is nothing to extract.
```

## Files to Modify

- `memory_extraction.h` — add Provider, Backend, Session members
- `memory_extraction.cpp` — replace HttpClient with backend generate

## Files Unchanged

- `rag.h` / `rag.cpp` — set_fact and store_memory already exist
- `rag/sqlite_backend.cpp` / `rag/postgresql_backend.cpp` — no changes
- `frontend.cpp` — injection stays as-is
- `config.h` / `config.cpp` — extraction config fields stay the same
