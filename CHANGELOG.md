# Changelog

All notable changes to Shepherd will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-11-02

### Added
- Extended thinking support for compatible models (Claude Sonnet 4.5, Gemini 2.0 Flash Thinking)
- Warmup control configuration options
- Tool reasoning display: assistant content shown before tool calls
- Comprehensive model definitions in models.cpp

### Changed
- CLI output formatting improvements:
  - Multi-line assistant responses and tool results now indented
  - Tool call indicator changed from `<` to `*`
  - Removed extra blank line before prompts
  - First line prefixed, subsequent lines indented for better readability
- Refactored Anthropic backend for improved error handling
- Enhanced Gemini backend integration
- Improved OpenAI backend with better model support
- Enhanced HTTP client retry logic

### Removed
- Grok backend support

## [2.0.0] - 2025-11-01

### Major Architectural Changes

#### Backend-First Message Transaction Pattern
- **Breaking Change**: Completely redesigned how messages are added to the conversation context
- **v1 Behavior**: `ContextManager` immediately added messages to local state, then sent to backend
  - Risk: Local state could become inconsistent if backend rejected the message
  - Token counts were estimated before backend confirmation
- **v2 Behavior**: `Session` sends message to backend FIRST, backend adds to session on success only
  - Transactional pattern ensures backend and local state stay synchronized
  - Backend sets accurate token counts from actual API responses
  - Messages only enter session.messages[] after successful backend acceptance
- **Implementation**: `Response backend->add_message(Session&, ...)` pattern
  - Backend is now responsible for updating `session.messages` on success
  - Allows backends to set precise token counts rather than estimates
  - Eliminates race conditions between context manager and backend state

#### Session-Based Architecture
- Replaced `ContextManager` with lightweight `Session` class
- Session is now the single source of truth for conversation state
- Simplified eviction logic moved into Session methods
- Removed complex `in_kv_cache` tracking flags from Message struct
- Direct delegation to backends for all operations

### Added

#### Code Organization
- **Separate CLI Module**: Extracted interactive/piped input handling into `cli.cpp`/`cli.h`
  - Previously all in monolithic `main.cpp` (91KB)
  - Improved maintainability and testability
  - Clean separation between CLI, server, and core logic

#### Message Tracking
- Added `last_user_message_index` and `last_assistant_message_index` to Session
- Tracks critical context that must be protected during eviction
- Enables accurate space reservation calculations

#### Auto-Eviction
- `Session::auto_evict` flag for proactive message eviction
- Enabled for API backends when user sets context_size < backend limit
- GPU backends continue to use reactive callback-based eviction

### Changed

#### Token Counting
- Token counts now authoritative from backend responses, not estimates
- `total_tokens` updated from API `usage.total_tokens` field
- Removed separate `current_token_count_` tracking in favor of backend authority
- Delta-based updates: `total_tokens += (usage.prompt_tokens - last_prompt_tokens)`

#### Eviction System
- Two-pass eviction strategy now in Session, not ContextManager
  - **Pass 1**: Evict complete conversation turns (USER → final ASSISTANT)
  - **Pass 2**: Evict mini-turns (ASSISTANT tool_call + TOOL result pairs)
- Protected message tracking prevents eviction of current turn context
- Simplified logic by removing `in_kv_cache` state management

#### Backend Interface
- Backends now receive `Session&` reference instead of managing separate context
- `Backend::add_message(Session&, type, content, ...)` returns Response
- Backend responsible for:
  - Formatting message for their API/format
  - Sending to model
  - Updating `session.messages` on success
  - Setting accurate token counts

### Removed

#### Deprecated Components
- **ContextManager class**: Replaced by Session + Backend delegation
- **`Message::in_kv_cache` field**: No longer needed with backend-first pattern
- **Separate token count tracking**: Unified under Session.total_tokens
- **`calculate_json_overhead()`**: Token accounting now backend-specific
- **`get_context_for_inference()`**: Backends format directly when sending

### Technical Details

#### Why This Change?

The v1 architecture had a fundamental ordering problem:

```cpp
// v1: ContextManager adds first, backend processes later
context_manager.add_message(msg);  // State updated immediately
backend.generate();                 // What if this fails?
```

This created race conditions where:
1. Local state thought message was added
2. Backend rejected it or counted tokens differently
3. State divergence between ContextManager and backend KV cache
4. Complex `in_kv_cache` tracking to reconcile differences

The v2 transactional pattern solves this:

```cpp
// v2: Backend processes first, adds to session on success only
Response resp = backend.add_message(session, msg);
// Backend has already updated session.messages if successful
// Token counts are accurate from backend response
// State is always consistent
```

#### Migration Notes

For developers extending Shepherd:

- **Custom Backends**: Must implement `add_message(Session&, ...)` pattern
  - Call `session.messages.push_back()` only after successful generation
  - Update `session.total_tokens` from actual usage
  - Return Response with success flag and error details

- **Eviction**: Use `Session::calculate_messages_to_evict()` and `Session::evict_messages()`
  - No need to track `in_kv_cache` state
  - Protected indices automatically preserved

- **Token Counting**: Trust backend token counts from API responses
  - Don't pre-calculate or override unless necessary
  - Delta updates prevent token count drift

## [1.0.0] - 2025-10-06

### Initial Release

- Multi-backend LLM system (llama.cpp, TensorRT-LLM, OpenAI, Anthropic, Gemini, Grok, Ollama)
- KV cache eviction with RAG archival
- Tool/function calling system
- Model Context Protocol (MCP) integration
- HTTP REST API server mode
- ContextManager-based conversation state
