# TensorRT-LLM KV Cache Eviction Implementation

## Overview

The TensorRT backend now supports automatic KV cache eviction with RAG archival and user notification, matching the functionality of the llama.cpp backend.

## Architecture

### Components

1. **Event Monitoring Thread** (`monitor_kv_events()`)
   - Runs in background
   - Polls `KVCacheEventManager` for events
   - Processes `KVCacheRemovedData` events when blocks are evicted

2. **TensorRTContextManager**
   - Tracks message-to-token position mappings
   - Provides `get_messages_in_token_range()` to map evicted tokens to messages
   - Maintains `message_token_positions_` vector for efficient lookup

3. **Eviction Handler** (`handle_kv_cache_removed()`)
   - Called when KV cache blocks are evicted
   - Estimates which messages were evicted
   - Calls `context_manager_->evict_messages_by_index()`
   - Shared logic archives to RAG and inserts NOTICE

## How It Works

### Event Flow

```
┌─────────────────────────────────────────────────────────────┐
│ TensorRT-LLM Executor                                       │
│                                                             │
│  1. KV cache fills up                                       │
│  2. Executor evicts old blocks automatically                │
│  3. Emits KVCacheRemovedData event                         │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Event Monitor Thread (monitor_kv_events)                   │
│                                                             │
│  4. Polls getLatestEvents() every 100ms                     │
│  5. Detects KVCacheRemovedData                             │
│  6. Calls handle_kv_cache_removed(block_hashes)            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Eviction Handler (handle_kv_cache_removed)                 │
│                                                             │
│  7. Estimates tokens evicted (blocks * 64)                 │
│  8. Calls calculate_messages_to_evict()                    │
│  9. Calls evict_messages_by_index()                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ ContextManager (evict_messages_by_index)                   │
│                                                             │
│ 10. Archives USER question + ASSISTANT answer to RAG       │
│ 11. Removes evicted messages from deque                    │
│ 12. Inserts NOTICE message:                                │
│     "NOTICE: Conversation regarding 'X...'                  │
│      has been moved to long-term memory."                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Next Generation Request                                     │
│                                                             │
│ 13. get_context_for_inference() includes NOTICE            │
│ 14. Prompt rendered with NOTICE visible to model           │
│ 15. Model sees: "NOTICE: ... Use search_memory tool"       │
└─────────────────────────────────────────────────────────────┘
```

### Key Differences from llama.cpp

| Aspect | llama.cpp | TensorRT-LLM |
|--------|-----------|--------------|
| **Trigger** | Synchronous callback before eviction | Async event after eviction |
| **Control** | We choose what to evict | TensorRT chooses |
| **Timing** | NOTICE in same generation | NOTICE in next request |
| **Mechanism** | Patch required | Built-in API |
| **Thread** | No extra thread | Background monitor thread |

### The Critical Insight

**We don't inject the NOTICE into the KV cache!** Instead:

1. NOTICE is added to `messages_` deque (our tracking layer)
2. Next `get_context_for_inference()` renders it as part of the prompt
3. That prompt is tokenized and submitted as a new Request
4. TensorRT processes it normally, adding NOTICE tokens to KV cache
5. Model sees the NOTICE in its context

This is the **same mechanism** llama.cpp uses - the difference is just timing.

## Implementation Details

### Message Position Tracking

```cpp
class TensorRTContextManager {
    // message_token_positions_[i] = cumulative tokens BEFORE message i
    std::vector<size_t> message_token_positions_;

    void add_message(const Message& message, int token_count) override {
        // Track position
        size_t tokens_before = 0;
        if (!messages_.empty()) {
            tokens_before = message_token_positions_.back() +
                           messages_.back().token_count;
        }
        message_token_positions_.push_back(tokens_before);

        // Add to base class
        ContextManager::add_message(message, token_count);
    }
};
```

### Event Monitoring Configuration

```cpp
// In initialize()
KvCacheConfig kvCacheConfig;
kvCacheConfig.setEnableBlockReuse(true);
kvCacheConfig.setEventBufferMaxSize(1000);  // CRITICAL: Enables events
config.setKvCacheConfig(kvCacheConfig);

// Get event manager
auto event_mgr_opt = executor->getKVCacheEventManager();
if (event_mgr_opt.has_value()) {
    event_manager_ = new std::shared_ptr<KVCacheEventManager>(*event_mgr_opt);

    // Start monitoring thread
    monitoring_events_ = true;
    kv_event_monitor_thread_ = std::thread(&TensorRTBackend::monitor_kv_events, this);
}
```

### Eviction Estimation

Currently uses a simple heuristic:
```cpp
// TensorRT typically uses 64-128 tokens per block
size_t estimated_tokens_evicted = block_hashes.size() * 64;
```

**Future Enhancement:** Track exact block→token mappings by monitoring `KVCacheStoredData` events.

## Example User Experience

```
User: What is quantum computing?
Assistant: [Long explanation about quantum computing...]

User: Tell me more about qubits.
Assistant: [Long explanation about qubits...]

... [many more turns] ...

[KV cache fills up, TensorRT evicts old blocks]
[Event detected, handler archives to RAG and adds NOTICE]

User: What was that thing you said about quantum computing earlier?
Assistant: I see there's a notice that our earlier conversation about
"What is quantum computing?..." has been moved to long-term memory.
Let me use the search_memory tool to retrieve it.

[Calls search_memory("quantum computing")]
[Retrieves from RAG]

Based on what I find in memory, I previously explained that...
```

The model naturally sees the NOTICE and knows to use `search_memory` to retrieve archived context.

## Configuration Options

### KV Cache Size

Control when eviction occurs by setting max tokens:

```cpp
KvCacheConfig kvCacheConfig;
kvCacheConfig.setMaxTokens(4096);  // Force small cache for testing
```

### Event Buffer Size

Control how many events are buffered:

```cpp
kvCacheConfig.setEventBufferMaxSize(1000);  // Default
```

Setting this to `0` disables event monitoring.

### Retention Priorities (Optional)

Prevent specific messages from being evicted:

```cpp
KvCacheRetentionConfig retention;
retention.addTokenRange(
    0,              // start
    system_tokens,  // end
    100             // priority (0-100, higher = keep longer)
);
request.setKvCacheRetentionConfig(retention);
```

## Thread Safety

- Event monitor thread runs independently
- Uses `block_map_mutex_` for concurrent access to `block_to_tokens_`
- Context manager operations are called from monitor thread
- No race conditions with main generation thread (events are queued)

## Performance Impact

- **Event polling**: 100ms timeout, minimal CPU usage
- **Event processing**: O(n) where n = number of messages (small)
- **Thread overhead**: Single background thread, negligible
- **Memory**: Event buffer ~1000 events, each ~100 bytes = ~100KB

## Limitations & Future Work

### Current Limitations

1. **Token estimation**: Uses 64 tokens/block heuristic instead of exact mapping
2. **Block mapping**: Doesn't track exact block→token→message mappings yet
3. **Timing**: NOTICE appears in next request, not same generation

### Future Enhancements

1. **Exact block tracking**:
   ```cpp
   // Monitor KVCacheStoredData events
   for (const auto& block : stored.blocks) {
       size_t start_token = calculate_token_position(block.tokens);
       size_t end_token = start_token + block.tokens.size();
       block_to_tokens_[block.blockHash] = {start_token, end_token};
   }
   ```

2. **Smarter eviction detection**:
   - Map evicted blocks to exact messages
   - Only evict complete conversation turns
   - Preserve message boundaries

3. **Retention policies**:
   - Auto-set high priority for system messages
   - Medium priority for recent turns
   - Low priority for old conversations

## Testing

### Test Eviction Manually

```bash
# Build with TensorRT enabled
cmake -DENABLE_TENSORRT=ON ..
make

# Run with small KV cache (forces eviction)
./shepherd --backend tensorrt \
           --model /path/to/engine \
           --max-context 512  # Small context forces eviction

# Have a long conversation to trigger eviction
# Watch logs for: "KV cache eviction detected"
```

### Verify NOTICE Messages

```bash
# Enable debug logging
./shepherd --backend tensorrt --model /path/to/engine --debug

# Look for logs:
# - "KV cache eviction detected: N blocks removed"
# - "Estimated X tokens evicted"
# - "Successfully evicted N messages, added eviction notice"
# - "Remaining tokens: X/Y"
```

## Comparison with llama.cpp

Both backends now support identical functionality:

✅ KV cache eviction detection
✅ RAG archival of evicted conversations
✅ NOTICE messages visible to model
✅ `search_memory` tool integration
✅ Automatic context management

The only difference is the underlying mechanism (callback vs events) and timing (immediate vs next-request).

## Files Modified

- `backends/tensorrt.h` - Added event monitoring infrastructure
- `backends/tensorrt.cpp` - Implemented monitor thread and handlers
- `context_manager.h` - Already had `evict_messages_by_index()` (shared)
- `context_manager.cpp` - Already had RAG archival + NOTICE logic (shared)

## References

- TensorRT KV Cache Events: `TENSORRT_KV_CACHE_EVENTS.md`
- llama.cpp Callback: `patches/llama.patch`
- Context Manager: `context_manager.cpp:257-344`