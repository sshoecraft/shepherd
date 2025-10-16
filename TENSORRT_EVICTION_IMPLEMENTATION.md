# TensorRT-LLM KV Cache Eviction Implementation

## Overview

The TensorRT backend supports reactive KV cache eviction with precise message tracking and RAG archival. This implementation leverages TensorRT's guaranteed LRU eviction behavior to precisely identify which messages were evicted without complex state tracking.

## Design Principles

**KISS (Keep It Simple, Stupid)**: The implementation follows a simple, reactive approach:
- TensorRT evicts blocks using LRU (Least Recently Used)
- We get notified after eviction happens
- We use token counting to precisely identify evicted messages
- We archive complete turns to RAG
- We track orphaned user questions for later completion

**No Proactive Management**: We don't try to control what TensorRT evicts - this would require expensive VRAM updates and complex priority management. Instead, we react to eviction events and maintain coherence.

## Architecture

### Key Components

1. **Token Position Tracking** (`TensorRTContextManager::message_token_positions_`)
   - Tracks cumulative token count before each message
   - Enables precise mapping of token ranges to messages
   - Updated automatically when messages are added

2. **Orphaned Question Tracking** (`TensorRTBackend::open_user_question_`)
   - Stores user messages that don't yet have assistant responses
   - Allows RAG archival when assistant response is later evicted
   - Single `std::optional<Message>` - only one open question at a time

3. **Retention Priorities** (configured per-request)
   - System message: Priority 100 (protected from eviction)
   - All other messages: Priority 35 (LRU eviction applies)
   - Static priorities - set once, never updated

4. **Event Monitoring Thread** (`monitor_kv_events()`)
   - Background thread polls `KVCacheEventManager`
   - Processes `KVCacheRemovedData` events
   - Calls eviction handler when blocks are removed

5. **Eviction Handler** (`handle_kv_cache_removed()`)
   - Calculates exact evicted messages using token counting
   - Scans for complete user→assistant turns
   - Archives turns to RAG
   - Removes evicted messages from tracking

## How It Works

### Eviction Flow

```
┌─────────────────────────────────────────────────────────────┐
│ TensorRT-LLM Executor                                       │
│                                                             │
│  1. KV cache fills up                                       │
│  2. Executor evicts oldest blocks (LRU, priority 35)        │
│  3. Emits KVCacheRemovedData event with block hashes       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Event Monitor Thread (monitor_kv_events)                   │
│                                                             │
│  4. Polls getLatestEvents() every 100ms                     │
│  5. Detects KVCacheRemovedData event                       │
│  6. Calls handle_kv_cache_removed(block_hashes)            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Eviction Handler (handle_kv_cache_removed)                 │
│                                                             │
│  7. Calculate tokens removed: N blocks × 64 tokens/block   │
│  8. Use LRU + token counting to identify messages 1-X      │
│  9. Check open_user_question_ for matching assistant      │
│ 10. Scan evicted range for user→assistant pairs            │
│ 11. Scan remaining messages for completing assistants      │
│ 12. Archive complete turns to RAG                          │
│ 13. Remove evicted messages from deque                     │
└─────────────────────────────────────────────────────────────┘
```

### Precise Message Identification

TensorRT uses **guaranteed LRU eviction** within each priority level. This means:

1. System message (priority 100) is never evicted
2. Other messages (priority 35) are evicted oldest-first
3. We can precisely calculate which messages were evicted:

```cpp
// Calculate tokens removed
size_t tokens_removed = num_blocks × 64 tokens_per_block;

// Sum token counts from oldest messages
size_t token_sum = 0;
for (size_t i = 1; i < messages.size(); ++i) {  // Skip system at 0
    token_sum += messages[i].token_count;
    if (token_sum >= tokens_removed) {
        // Messages 1 through i were evicted
        last_evicted_msg = i;
        break;
    }
}
```

**No heuristics. No estimation. Exact calculation.**

### RAG Archival Strategy

The handler scans for complete conversation turns in three phases:

**Phase 1: Complete orphaned turn**
```cpp
if (open_user_question_.has_value()) {
    // Check if evicted range contains the assistant response
    if (found assistant in messages 1-X) {
        archive_turn(open_user_question_, assistant);
        open_user_question_.reset();
    }
}
```

**Phase 2: Complete turns in evicted range**
```cpp
for each user message in evicted range (1-X) {
    if (assistant response also in 1-X) {
        archive_turn(user, assistant);
    }
}
```

**Phase 3: Complete turns spanning eviction boundary**
```cpp
for each user message in evicted range (1-X) {
    if (assistant response in remaining messages X+1 onwards) {
        archive_turn(user, assistant);
    }
}
```

**Phase 4: Store new orphaned questions**
```cpp
if (user message found with no assistant anywhere) {
    open_user_question_ = user_message;
}
```

This maximizes RAG archival - we preserve every complete turn we can, even when they span the eviction boundary.

## Implementation Details

### Retention Priority Configuration

```cpp
// In generate() method, before creating Request
size_t system_msg_tokens = trt_ctx_mgr->get_tokens_before_message(1);

std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig> token_ranges;
if (system_msg_tokens > 0) {
    token_ranges.push_back(
        tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(
            0,                    // start token
            system_msg_tokens,    // end token (exclusive)
            100                   // priority - never evict
        )
    );
}

tle::KvCacheRetentionConfig retention(
    token_ranges,
    35  // decodeRetentionPriority - everything else
);

tle::Request request(
    input_tokens,
    max_tokens,
    true,  // streaming
    samplingConfig,
    outputConfig,
    /* ... many nullopt parameters ... */
    retention  // kvCacheRetentionConfig
);
```

### Message Position Tracking

```cpp
void TensorRTContextManager::add_message(const Message& message) {
    // Track cumulative token position before this message
    size_t tokens_before = 0;
    if (!messages_.empty()) {
        tokens_before = message_token_positions_.back() +
                       messages_.back().token_count;
    }
    message_token_positions_.push_back(tokens_before);

    // Call base class
    ContextManager::add_message(message);
}
```

### Event Monitoring Setup

```cpp
// In initialize()
tle::KvCacheConfig kvCacheConfig;
kvCacheConfig.setEnableBlockReuse(true);
kvCacheConfig.setEventBufferMaxSize(1000);  // Enable event buffering
config.setKvCacheConfig(kvCacheConfig);

// Create executor
auto* executor = new tle::Executor(model_path, tle::ModelType::kDECODER_ONLY, config);

// Get event manager
auto event_mgr_opt = executor->getKVCacheEventManager();
if (event_mgr_opt.has_value()) {
    event_manager_ = new std::shared_ptr<tle::KVCacheEventManager>(*event_mgr_opt);

    // Start monitoring thread
    monitoring_events_ = true;
    kv_event_monitor_thread_ = std::thread(&TensorRTBackend::monitor_kv_events, this);
}
```

## Corner Cases

### User Question Evicted During Tool Execution

**Scenario**: User asks "Read every file in this project", model starts executing tool calls, context fills up, user message gets evicted before assistant response is generated.

**Result**:
- Context becomes: [System message] + [Tool results]
- No user question in context
- Model continues generating based on available context (tool results + system)

**Why this is acceptable**:
- This is a universal problem for any LLM system with KV cache eviction
- The model likely generates a reasonable response based on tool results
- This corner case is rare in practice
- Over-engineering a solution (e.g., canceling generation) would be worse

**What we do**:
- Let generation continue
- Remove all evicted messages from our tracking
- Deque remains coherent

## Thread Safety

- **Event monitor thread**: Runs independently in background
- **Mutex protection**: `block_map_mutex_` protects concurrent access
- **Message deque**: Only accessed from monitor thread during eviction
- **No race conditions**: TensorRT queues events, no concurrent modification

## Performance

- **Event polling**: 100ms timeout, minimal CPU overhead
- **Token counting**: O(n) where n = number of messages (typically < 100)
- **Message removal**: O(n) for deque operations
- **Thread overhead**: Single background thread, ~1MB stack
- **Memory**: Event buffer ~100KB, position tracking ~1KB per 100 messages

**Total impact**: Negligible

## Configuration

### Force Eviction for Testing

```bash
./shepherd --backend tensorrt \
           --model /path/to/engine \
           --max-context 512  # Small context forces frequent eviction
```

### Enable Debug Logging

```bash
./shepherd --backend tensorrt --model /path/to/engine --debug

# Watch for logs:
# - "KV cache eviction: N blocks removed (~X tokens)"
# - "Identified messages 1-Y as evicted"
# - "Found complete turn ... archiving to RAG"
# - "User message has no assistant response yet, storing as orphaned"
# - "Removed Y messages from context, Z remaining"
```

## Files Modified

- `backends/tensorrt.h`:
  - Added `message_token_positions_` to TensorRTContextManager
  - Added `open_user_question_` member to TensorRTBackend
  - Added `remove_message_at_index()` helper method

- `backends/tensorrt.cpp`:
  - Implemented retention priority configuration in `generate()`
  - Refactored `handle_kv_cache_removed()` with token counting logic
  - Added RAG archival for complete turns
  - Added orphaned question tracking

## Key Insights

### Why No Coherence Check?

Because **token counting makes the deque automatically coherent**:

1. TensorRT evicts messages 1 through X (we calculate this precisely)
2. We remove messages 1 through X from our deque
3. Our deque now exactly matches TensorRT's cache

There's no "orphaned tool results" or "missing user messages" because we removed exactly what TensorRT removed. The mapping is one-to-one.

### Why No Priority Updates?

Because **TensorRT doesn't support dynamic priority updates**:

- Block priorities are set when blocks are created
- They're baked into the KV cache state
- There's no API to retroactively change them
- Any attempt would require expensive VRAM manipulation

So we use **static priorities**: system=100 (never evict), everything else=35 (LRU).

### Why Single open_user_question_?

Because **normal conversation flow only has one open question at a time**:

```
User message 1
Assistant response 1  ← completes
User message 2
Assistant response 2  ← completes
```

If we somehow encounter a second user message while `open_user_question_` has data, it means something went wrong. We could:
- Assert/error (something broke)
- Overwrite with new question (lose old one)

Currently we overwrite, as this corner case shouldn't happen in practice.

## Comparison with llama.cpp

| Aspect | llama.cpp | TensorRT |
|--------|-----------|----------|
| **Mechanism** | Callback before eviction | Event after eviction |
| **Control** | We choose what to evict | TensorRT chooses (LRU) |
| **Accuracy** | Exact (we control eviction) | Exact (token counting + LRU guarantee) |
| **RAG archival** | Complete turns only | Complete turns + spanning boundary |
| **Orphaned questions** | No special handling | Tracked in `open_user_question_` |
| **Priority system** | N/A (we control eviction) | Static priorities (system=100) |
| **Thread overhead** | None (callback on main thread) | Background event monitor thread |

Both achieve the same goal with different approaches.

## Testing

### Basic Eviction Test

```bash
# Build
cmake -DENABLE_TENSORRT=ON -B build
cmake --build build -j8

# Run with small context
./build/shepherd --backend tensorrt \
                 --model /path/to/llama-3.1-8B-instruct-engine \
                 --max-context 1024

# Have a conversation with multiple long turns
# Watch debug logs for eviction events
```

### Verify RAG Archival

```bash
# After eviction, check RAG database
sqlite3 ~/.shepherd/rag.db "SELECT question, answer FROM turns ORDER BY timestamp DESC LIMIT 5;"

# Verify turns were archived correctly
```

### Test Orphaned Questions

```bash
# User: "Read every file in /huge/directory"
# [Model starts tool execution]
# [Context fills up before assistant response]
# [User message gets evicted]

# Check logs:
# - "User message has no assistant response yet, storing as orphaned"
# - "Removed N messages from context"

# On next eviction (if assistant response gets evicted):
# - "Found assistant response for orphaned user question, archiving to RAG"
```

## References

- TensorRT KV Cache Events API: `TENSORRT_KV_CACHE_EVENTS.md`
- LRU Eviction Policy: `include/tensorrt_llm/batch_manager/evictionPolicy.h`
- RAG Manager: `rag.h`, `rag.cpp`
- Context Manager: `context_manager.h`, `context_manager.cpp`
