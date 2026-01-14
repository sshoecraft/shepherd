# Token-Level Prefix Caching Architecture

## Document Purpose

This document describes the architectural changes made to shepherd's KV cache management system to fix the "tart" corruption bug affecting GPT-OSS (harmony models) in server mode.

## Problem Statement

### Symptoms
- GPT-OSS model produced corrupted output like "Ctart", "Btart" on Turn 2+ of multi-turn conversations
- Only occurred in shepherd's server mode (OpenAI-compatible API)
- Did NOT occur with llama-server or vLLM (0 errors)
- Did NOT occur in CLI mode

### Original Architecture (Message-Level Tracking)

The original `generate_from_session()` in `backends/llamacpp.cpp` used **message-level prefix caching**:

```cpp
// OLD: Track messages with per-message token counts
Session backend_session;  // Contains: {messages: [{content, tokens}, ...]}

// Compare messages to find prefix
for (size_t i = 0; i < session.messages.size(); i++) {
    if (cached_msg.content == session_msg.content) {
        matching_prefix++;
    }
}

// Calculate token position from message token counts
int clear_from_pos = 0;
for (size_t i = 0; i < expected_backend_count; i++) {
    clear_from_pos += backend_session.messages[i].tokens;
}
```

### Root Cause

For harmony models (GPT-OSS), the assistant response contains multiple channels:
```
<|channel|>analysis<|message|>thinking content here...
<|channel|>final<|message|>C<|end|>
```

The bug occurred because:

1. **During generation**: The full response (analysis + final) was generated and stored in KV cache
2. **Content extraction**: `extract_content()` correctly extracted only "C" from the final channel
3. **Storage mismatch**: We stored `{content: "C", tokens: 150}` - the extracted content but the FULL token count
4. **Re-rendering**: On the next request, rendering "C" through the template produced ~10 tokens, not 150
5. **Position corruption**: The calculated `clear_from_pos` was wrong, leading to KV cache corruption

The fundamental flaw: **message content didn't match what was actually in the KV cache**.

## New Architecture (Token-Level Tracking)

### Design Principle

Instead of tracking messages with token counts, track the **actual tokens** in the KV cache. This eliminates any possibility of content/token mismatch.

### New Data Structure

```cpp
// backends/llamacpp.h, line 150-153
class LlamaCppBackend {
    // ...

    // Token-level KV cache tracking (for accurate prefix caching)
    // This stores the actual tokens that are in the KV cache, enabling
    // precise prefix matching regardless of how content was extracted/stored
    std::vector<llama_token> kv_cache_tokens;
};
```

### New Flow in generate_from_session()

Location: `backends/llamacpp.cpp`, lines 1808-1960

```cpp
void LlamaCppBackend::generate_from_session(Session& session, ...) {
    // Step 1: Build message list for rendering
    std::vector<Message> all_messages;
    if (!session.system_message.empty()) {
        std::string formatted_system = chat_template->format_system_message(...);
        all_messages.push_back(Message(Message::SYSTEM, formatted_system, 0));
    }
    for (const auto& msg : session.messages) {
        all_messages.push_back(msg);
    }

    // Step 2: Render full conversation WITHOUT generation prompt
    // (generate() adds the generation prompt itself)
    std::string rendered = chat_template->format_conversation(all_messages, session.tools, false);

    // Step 3: Tokenize the rendered conversation
    std::vector<llama_token> new_tokens(rendered.length() + 256);
    int n_new_tokens = llama_tokenize(vocab, rendered.c_str(), ...);
    new_tokens.resize(n_new_tokens);

    // Step 4: Find matching prefix (simple array comparison)
    size_t prefix_len = 0;
    size_t max_compare = std::min(kv_cache_tokens.size(), new_tokens.size());
    while (prefix_len < max_compare &&
           kv_cache_tokens[prefix_len] == new_tokens[prefix_len]) {
        prefix_len++;
    }

    // Step 5: Clear KV cache from divergence point
    // CRITICAL: Check ACTUAL KV cache size, not just kv_cache_tokens.size()
    // After generation, KV cache has untracked tokens (gen_prompt + response + closing)
    int actual_kv_tokens = get_context_token_count();
    if (prefix_len < static_cast<size_t>(actual_kv_tokens)) {
        llama_memory_seq_rm(mem, 0, prefix_len, -1);
    }
    if (prefix_len < kv_cache_tokens.size()) {
        kv_cache_tokens.resize(prefix_len);
    }

    // Step 6: Decode delta tokens (only new ones after prefix)
    if (delta_tokens > 0) {
        for (size_t i = prefix_len; i < new_tokens.size(); i += n_batch) {
            int batch_size = std::min(n_batch, static_cast<int>(new_tokens.size() - i));
            llama_batch batch = llama_batch_get_one(new_tokens.data() + i, batch_size);
            llama_decode(ctx, batch);
        }
    }

    // Step 7: Update kv_cache_tokens (prompt tokens only)
    kv_cache_tokens = new_tokens;

    // ... then call generate() which adds gen_prompt and runs inference
}
```

### Critical Design Decision: Not Tracking Generated Tokens

After generation completes, we do **NOT** update `kv_cache_tokens` with the generated assistant tokens. This is intentional.

Location: `backends/llamacpp.cpp`, lines 1991-2013

```cpp
// NOTE: We do NOT update kv_cache_tokens here because:
// - For harmony models: KV cache has analysis+final, re-rendering produces only final
// - For all models: KV cache has generation_prompt + generated + closing tokens
// - Re-rendering produces template-wrapped content which may differ in tokens
//
// Instead, kv_cache_tokens stays as the PROMPT tokens only. On next request:
// - System+User tokens will match (prefix cache hit for prompt)
// - Assistant tokens won't be in kv_cache_tokens, so they'll be re-decoded
// - This is correct behavior: assistant response is re-decoded through template
```

This means:
- `kv_cache_tokens` only contains prompt tokens (system + user messages)
- Generated assistant tokens remain in KV cache but aren't tracked
- On next request, we detect the divergence and clear the untracked tokens

### Why Check actual_kv_tokens?

The key bug fix is at line 1882-1886:

```cpp
int actual_kv_tokens = get_context_token_count();  // Query real KV cache state
if (prefix_len < static_cast<size_t>(actual_kv_tokens)) {
    llama_memory_seq_rm(mem, 0, prefix_len, -1);
}
```

After Request 1:
- `kv_cache_tokens` = 100 (prompt tokens)
- Actual KV cache = 150 (prompt + gen_prompt + generated + closing)

On Request 2:
- `new_tokens` = 120 (prompt + new user message)
- `prefix_len` = 100 (first 100 match)
- `kv_cache_tokens.size()` = 100

If we only checked `prefix_len < kv_cache_tokens.size()`:
- 100 < 100 = FALSE
- We wouldn't clear anything!
- New tokens would append at position 150, corrupting context

By checking `actual_kv_tokens`:
- 100 < 150 = TRUE
- We clear from position 100 to end
- New tokens decode correctly starting at position 100

## Performance Characteristics

### Benchmark Results (from planning phase)

| Turns | Tokens | Tokenize Time | Compare Time |
|-------|--------|---------------|--------------|
| 10    | 1.5K   | 2.7ms         | 0.02ms       |
| 50    | 7.7K   | 7.5ms         | 0.09ms       |
| 100   | 15.5K  | 15.0ms        | 0.17ms       |
| 200   | 31.0K  | 30.0ms        | 0.38ms       |

At 30 t/s generation (~33ms/token), tokenization overhead is less than 1 token. Negligible.

### Trade-offs

**Benefits:**
1. Fixes harmony bug - no more content/token count mismatch
2. Works for ALL models - no special cases needed
3. Simpler conceptual model - track what's actually in cache
4. Proven approach - vLLM uses similar token-level tracking

**Costs:**
1. Re-tokenizes full conversation each request (~7-30ms)
2. Re-decodes assistant response each turn (can't cache generated content)
3. Memory for storing token vector (negligible)

## Test Results

### GPT-OSS MMLU Benchmark

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Errors | 17-20/100  | 0/100     |
| Accuracy | N/A (corrupted) | 72-79% |

### O(1) Response Time Verification

With GLM-4-9B (30 sequential requests):
```
First 10 requests avg: 51.2ms (includes warmup)
Last 10 requests avg:  33.4ms
Ratio: 0.65x (stable/improving)
```

Response times remain constant regardless of request count, confirming O(1) behavior.

## Files Modified

### backends/llamacpp.h
- Line 150-153: Added `std::vector<llama_token> kv_cache_tokens` member

### backends/llamacpp.cpp
- Lines 1808-1960: Rewrote `generate_from_session()` with token-level prefix caching
- Lines 1991-2013: Added documentation explaining why we don't track generated tokens

## Bug Fixes During Implementation

### Qwen Template tool_calls Crash (FIXED)

**Problem:** Qwen3-Coder template crashed with "Can only get item pairs from a mapping" during capability probing.

**Root Cause:** The Qwen template does:
```jinja
{%- if tool_call.function is defined %}
    {%- set tool_call = tool_call.function %}  <-- Reassigns tool_call!
{%- endif %}
{%- for args_name, args_value in tool_call.arguments|items %}
```

After reassignment, `tool_call.arguments` points to `function.arguments`, which was being set as a JSON string, not a dict.

**Fix:** In `chat_template.cpp`, parse the arguments JSON string and set the resulting object on BOTH `tc_obj.arguments` AND `func_obj.arguments`:

```cpp
// chat_template.cpp lines 543-561
auto parsed = nlohmann::ordered_json::parse(args_str);
if (parsed.is_object()) {
    args_obj = minja::Value(parsed);
    func_obj.set("arguments", args_obj);  // Set on function too!
}
tc_obj.set("arguments", args_obj);
tc_obj.set("function", func_obj);
```

**Result:** `supports_tool_calls: true` for Qwen templates.

---

## Known Issues / Regressions Introduced

### DeepSeek Performance and Output Regression (CRITICAL)

**This refactor introduced severe regressions** affecting DeepSeek-R1-Distill-Llama-70B (and possibly other models):

**Before refactor:**
- Speed: 11 t/s
- Output: Correct

**After refactor:**
- Speed: 2.03 t/s (5x slower!)
- Output: Garbage (backslash sequences like `\\\\\\\\...`)

Both the performance regression and the output corruption occur on the FIRST request, indicating fundamental issues with the new code path.

Possible causes:
1. Something in how `format_conversation()` is being called
2. The message list construction (lines 1826-1838)
3. Unnecessary re-tokenization/re-decoding overhead
4. Interaction between the new flow and certain chat templates

**The refactor is NOT complete until both regressions are resolved.**

## API Reference

### get_context_token_count()

Location: `backends/llamacpp.cpp`, lines 891-907

```cpp
int LlamaCppBackend::get_context_token_count() const {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_pos actual_max_pos = llama_memory_seq_pos_max(mem, 0);
    return (actual_max_pos >= 0) ? (actual_max_pos + 1) : 0;
}
```

Returns the actual number of tokens in the KV cache by querying llama.cpp's internal state. This is the source of truth for what's in the cache.

### llama_memory_seq_rm()

```cpp
LLAMA_API bool llama_memory_seq_rm(
    llama_memory_t mem,
    llama_seq_id seq_id,  // Usually 0 for single-sequence
    llama_pos p0,         // Start position (inclusive)
    llama_pos p1);        // End position (-1 = to end)
```

Removes tokens from the KV cache. Used to clear stale content when conversations diverge.

## Diagram

```
Request N:
┌─────────────────────────────────────────────────────────────┐
│ 1. Render conversation → rendered string                     │
│ 2. Tokenize → new_tokens[]                                   │
│ 3. Compare new_tokens with kv_cache_tokens                   │
│ 4. Find prefix_len (matching portion)                        │
│ 5. Check actual_kv_tokens (real KV cache size)               │
│ 6. If divergence: llama_memory_seq_rm(prefix_len, -1)        │
│ 7. Decode new_tokens[prefix_len:] into KV cache              │
│ 8. kv_cache_tokens = new_tokens                              │
│ 9. generate() → adds gen_prompt, samples response            │
│ 10. KV cache now has: prompt + gen_prompt + response         │
│ 11. kv_cache_tokens still = prompt only (intentional)        │
└─────────────────────────────────────────────────────────────┘

Request N+1:
┌─────────────────────────────────────────────────────────────┐
│ 1. Render with previous assistant response → new_tokens      │
│ 2. Compare: system+user prefix matches kv_cache_tokens       │
│ 3. prefix_len = length of prompt                             │
│ 4. actual_kv = prompt + gen_prompt + response (larger!)      │
│ 5. prefix_len < actual_kv → CLEAR from prefix_len            │
│ 6. Re-decode assistant message through template              │
│ 7. Continue with new user message                            │
└─────────────────────────────────────────────────────────────┘
```

## Current Status

**NOT READY FOR USE.** The refactor:
- Fixes the GPT-OSS "tart" corruption (verified with MMLU benchmark: 0 errors)
- Introduces a regression breaking DeepSeek (and possibly other models)

The regression must be debugged and fixed before this architecture can be deployed.

## Next Steps

1. Debug why DeepSeek produces garbage output after the refactor
2. Compare the rendered output between old and new code paths
3. Test with other models (Llama, Qwen) to identify scope of regression
4. Only after all models work correctly: merge the changes
