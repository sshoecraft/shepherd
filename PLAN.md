# Rework Memory Extraction + Injection Pipeline

## Context
Architecture: Shepherd API server (`memory: true`) handles RAG. CLI client (`memory: false`) connects to it. GLM-4 9B runs as extraction model.

Problems found:
1. Extraction prompt extracted assistant self-descriptions — **already fixed** in `memory_extraction.cpp`
2. Injection returns garbage — old entries like "name is not provided" outrank actual facts because search is pure text relevance with no recency
3. Raw `archive_turn` during eviction is redundant — extraction already captured anything valuable
4. Facts and context are mixed in one table with no distinction — injection can't prioritize properly

## Changes

### 1. Disable archive_turn during eviction
- `session.cpp` line 209: disable `RAGManager::archive_turn()` call
- `backends/tensorrt.cpp` lines 1739, 1759, 1774: disable archive_turn calls
- May be removed entirely in a future session

### 2. Add time-based relevancy to search
- `rag/sqlite_backend.cpp`: incorporate timestamp into scoring (combine BM25 with recency)
- `rag/postgresql_backend.cpp`: incorporate timestamp into `ts_rank_cd` scoring (combine with recency)
- Newer entries about the same topic should outrank older ones

### 3. Rework extraction to produce facts AND context
- Keep current HTTP-based extraction approach
- Change extraction prompt to request JSON output with two sections:
  ```json
  {
    "facts": {"name": "Steve", "location": "Texas"},
    "context": [
      {"q": "What treatment was discussed?", "a": "Cream for skin irritation"}
    ]
  }
  ```
- Empty sections when nothing to extract: `{"facts": {}, "context": []}`
- **Facts**: key-value pairs → stored to `facts` table via `RAGManager::set_fact()` (INSERT OR REPLACE — newer value always wins, no duplicates). No get_fact/search before storing — just store directly.
- **Context**: Q/A pairs → stored to context table (currently `conversations`)
- Rework `parse_and_store_facts()` to parse JSON and route each type to the right storage
- **Future work** (separate session): refactor extraction to use APIToolAdapter with a proper provider, giving the extraction model direct tool access (set_fact, store_context). Better architecture, but requires models that support tool calling.

### 4. Rename database table
- `conversations` → `context` (schema migration, idempotent ALTER TABLE RENAME)
- `facts` table stays as-is
- Update all SQL references in both SQLite and PostgreSQL backends

### 5. Rework injection to present facts and context separately
- `frontend.cpp` `enrich_with_rag_context()`: two retrieval paths
  - Search `facts` table for user's facts → present as `[facts: ...]`
  - Search `context` table with time-based relevancy → present as `[context: ...]`
- Facts appear first, context second — model clearly sees the difference

### Already done
- `memory_extraction.cpp`: Updated EXTRACTION_SYSTEM_PROMPT
- `CMakeLists.txt`: Version bumped to 2.31.1

## Files to modify
- `memory_extraction.cpp` — new extraction prompt format, new parser
- `memory_extraction.h` — if struct changes needed
- `rag/sqlite_backend.cpp` — table rename, time-based search
- `rag/sqlite_backend.h` — if interface changes
- `rag/postgresql_backend.cpp` — table rename, time-based search
- `rag/postgresql_backend.h` — if interface changes
- `rag/database_backend.h` — if interface changes
- `frontend.cpp` — reworked injection with separate facts/context
- `session.cpp` — disable archive_turn
- `backends/tensorrt.cpp` — disable archive_turn
- `rag.cpp` / `rag.h` — any wrapper changes
- `docs/memory_extraction.md` — update docs
- `docs/rag.md` / `docs/rag_context.md` — update docs

## Verification
- Rebuild and start server with extraction configured
- Send "my name is Steve and I live in Texas" → check extraction stores facts via set_fact, not into context table
- Send "we discussed using cream for itching" → check extraction stores context Q/A into context table
- New session: "what is my name" → injection should show `[facts: name=Steve, location=Texas]` and model answers correctly
- Send "call me Skippy" → check set_fact overwrites name=Steve with name=Skippy (INSERT OR REPLACE)
- New session: "what is my name" → should say Skippy, not Steve
- Verify archive_turn is disabled (no raw turns archived during eviction)

## Version
2.32.0 (new feature: separated facts/context extraction and injection)
