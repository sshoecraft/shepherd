# TensorRT-LLM KV Cache Event System

## Overview

Unlike llama.cpp which requires a patch to add a callback, **TensorRT-LLM has a built-in event system** for monitoring KV cache operations! This is a more sophisticated approach than the callback mechanism.

## Key Difference from llama.cpp

### llama.cpp Approach (Requires Patch)
- **Synchronous callback**: When KV cache is full, llama.cpp calls a callback function
- **Blocking**: The callback must free space immediately before llama.cpp continues
- **Patched in**: We added `kv_need_space_callback` via `patches/llama.patch`

### TensorRT-LLM Approach (Built-in)
- **Event-based**: TensorRT-LLM emits events as KV cache blocks are created/removed/updated
- **Asynchronous**: Application polls for events and can react accordingly
- **Native**: No patching required, it's part of the official API

## TensorRT-LLM KV Cache Event API

### Event Types

Located in `tensorrt_llm/executor/executor.h`:

```cpp
// Event data is a variant of these types:
struct KVCacheCreatedData {
    std::vector<SizeType32> numBlocksPerCacheLevel;
};

struct KVCacheStoredData {
    std::optional<IdType> parentHash;
    std::vector<KVCacheStoredBlockData> blocks;
};

struct KVCacheRemovedData {
    std::vector<IdType> blockHashes;  // Blocks being evicted
};

struct KVCacheUpdatedData {
    IdType blockHash;
    std::optional<KVCacheEventDiff<SizeType32>> cacheLevel;
    std::optional<KVCacheEventDiff<SizeType32>> priority;
};

using KVCacheEventData = std::variant<
    KVCacheCreatedData,
    KVCacheStoredData,
    KVCacheRemovedData,
    KVCacheUpdatedData
>;
```

### Event Structure

```cpp
struct KVCacheEvent {
    IdType eventId;              // Unique event ID
    KVCacheEventData data;       // Event-specific data
    SizeType32 windowSize;       // Sliding window size
    std::optional<SizeType32> attentionDpRank;
};
```

### KVCacheEventManager Class

```cpp
class KVCacheEventManager {
public:
    /// Get the latest KV Cache events
    /// @param timeout Max time to wait for new events. If nullopt, blocks until events available
    std::deque<KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt
    );
};
```

### Accessing the Event Manager

From the Executor:

```cpp
class Executor {
    // ...
    std::optional<std::shared_ptr<KVCacheEventManager>> getKVCacheEventManager() const;
};
```

## Configuration

### Enable Event Monitoring

In `KvCacheConfig`:

```cpp
KvCacheConfig kvCacheConfig;
kvCacheConfig.setEnableBlockReuse(true);
kvCacheConfig.setEventBufferMaxSize(1000);  // Enable event buffering!

ExecutorConfig config;
config.setKvCacheConfig(kvCacheConfig);
```

**Key**: Setting `eventBufferMaxSize > 0` enables event collection.

### KV Cache Retention (Advanced)

TensorRT-LLM also supports fine-grained KV cache retention policies per request:

```cpp
KvCacheRetentionConfig retentionConfig;

// Set retention priority for different token ranges
retentionConfig.getTokenRangeRetentionConfigs().push_back(
    KvCacheRetentionConfig::TokenRangeRetentionConfig(
        0,        // tokenStart
        100,      // tokenEnd
        90,       // priority (0-100, higher = less likely to evict)
        std::nullopt  // duration
    )
);

// Add to request
request.setKvCacheRetentionConfig(retentionConfig);
```

## Implementation Strategy for Shepherd

### Option 1: Event-Based Monitoring (Recommended)

**Pros:**
- No patching required
- Asynchronous - doesn't block inference
- Can track all KV cache operations
- Production-ready API

**Cons:**
- More complex - need event polling thread
- Reactive rather than proactive (gets notified AFTER eviction)

**Implementation:**

```cpp
class TensorRTBackend {
    // Monitor thread
    std::thread kv_event_monitor_;
    std::atomic<bool> monitoring_;

    void monitor_kv_events() {
        auto event_mgr = executor_->getKVCacheEventManager();
        if (!event_mgr.has_value()) return;

        while (monitoring_) {
            auto events = (*event_mgr)->getLatestEvents(
                std::chrono::milliseconds(100)
            );

            for (const auto& event : events) {
                if (std::holds_alternative<KVCacheRemovedData>(event.data)) {
                    const auto& removed = std::get<KVCacheRemovedData>(event.data);
                    LOG_INFO("KV cache evicted " +
                             std::to_string(removed.blockHashes.size()) + " blocks");

                    // TODO: Archive corresponding messages to RAG
                    // This requires mapping block hashes to messages
                }
            }
        }
    }
};
```

### Option 2: Retention Priority-Based (Simpler)

**Pros:**
- Declarative - set priorities and let TensorRT-LLM handle it
- No monitoring thread needed
- Works per-request

**Cons:**
- Less control over what gets evicted
- Can't archive to RAG automatically
- Need to set priorities correctly upfront

**Implementation:**

```cpp
void TensorRTBackend::generate(int max_tokens) {
    // Set retention config for this request
    KvCacheRetentionConfig retention;

    // System messages: highest priority (never evict)
    retention.getTokenRangeRetentionConfigs().push_back({
        0, system_tokens, 100  // priority 100
    });

    // Recent user messages: high priority
    retention.getTokenRangeRetentionConfigs().push_back({
        system_tokens, system_tokens + recent_tokens, 80
    });

    // Old messages: low priority (evict first)
    retention.setDecodeRetentionPriority(30);

    request.setKvCacheRetentionConfig(retention);
}
```

### Option 3: Hybrid Approach (Best)

1. **Use retention priorities** to guide TensorRT-LLM's eviction decisions
2. **Monitor events** to know what was evicted and archive to RAG
3. **Map block hashes to messages** to maintain consistency

## Challenges

### Block Hash to Message Mapping

TensorRT-LLM operates on **block hashes** (based on token sequences), while Shepherd tracks **messages**. We need to:

1. Track which token ranges correspond to which messages
2. When blocks are evicted, determine which messages they belonged to
3. Archive those messages to RAG

This is complex because:
- Multiple blocks can belong to one message
- Block reuse means same tokens may be shared across requests
- Token positions shift as context evolves

### Suggested Solution

```cpp
class TensorRTContextManager {
private:
    // Track message -> token range mapping
    struct MessageTokenRange {
        size_t message_index;
        size_t token_start;
        size_t token_end;
    };
    std::vector<MessageTokenRange> message_token_ranges_;

    // Track block hash -> token range mapping (from events)
    std::map<IdType, std::pair<size_t, size_t>> block_to_tokens_;

    void on_kv_cache_removed(const KVCacheRemovedData& removed) {
        std::set<size_t> affected_messages;

        for (IdType hash : removed.blockHashes) {
            if (block_to_tokens_.count(hash)) {
                auto [start, end] = block_to_tokens_[hash];

                // Find messages that overlap this token range
                for (const auto& mtr : message_token_ranges_) {
                    if (mtr.token_start < end && mtr.token_end > start) {
                        affected_messages.insert(mtr.message_index);
                    }
                }

                block_to_tokens_.erase(hash);
            }
        }

        // Archive affected messages to RAG
        for (size_t msg_idx : affected_messages) {
            archive_message_to_rag(messages_[msg_idx]);
        }
    }
};
```

## Next Steps

1. **Implement event monitoring thread** in TensorRTBackend
2. **Track block-to-message mappings** in TensorRTContextManager
3. **Enable event buffering** in ExecutorConfig
4. **Add RAG archiving** when blocks are evicted
5. **Test with limited KV cache** to trigger evictions

## References

- Event structures: `tensorrt_llm/executor/executor.h` lines 1665-1781
- KV cache config: `tensorrt_llm/executor/executor.h` lines 997-1090
- Retention config: `tensorrt_llm/executor/executor.h` lines 545-616
