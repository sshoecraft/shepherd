# Shepherd Client Eviction Test Plan

## Overview

This document provides a comprehensive manual test plan for verifying Shepherd CLIENT's eviction and RAG archival behavior when connecting to a Shepherd server.

**What We're Testing:**
- ✅ Client-side eviction triggers at configured context size
- ✅ Evicted messages are archived to RAG database
- ✅ RAG retrieval works (search_memory tool)
- ✅ Server never sees errors (client manages context)
- ✅ Conversation coherence after eviction

**Why Manual Testing:**
Shepherd is an interactive program that requires a terminal. Automated testing with piped input doesn't work because Shepherd expects TTY interaction.

---

## Prerequisites

1. **Shepherd Server Running:**
   ```bash
   # On server machine (192.168.1.166)
   ./shepherd --server --port 8000 \
     --backend llamacpp \
     --model /path/to/model.gguf \
     --context-size 98304
   ```

2. **Shepherd Client Binary Built:**
   ```bash
   cd /Users/steve/src/shepherd/build
   ./shepherd --help  # Verify it works
   ```

---

## Test Setup

Before each test, start Shepherd CLIENT with specific configuration:

```bash
cd /Users/steve/src/shepherd/build

# Test 1: Tiny context (2048 tokens)
./shepherd \
  --backend openai \
  --api-base http://192.168.1.166:8000/v1 \
  --model gpt-4 \
  --context-size 2048 \
  --rag-db /tmp/eviction_test_tiny.db

# Test 2: Small context (8192 tokens)
./shepherd \
  --backend openai \
  --api-base http://192.168.1.166:8000/v1 \
  --model gpt-4 \
  --context-size 8192 \
  --rag-db /tmp/eviction_test_small.db

# Test 3: Micro context (512 tokens) - rapid eviction
./shepherd \
  --backend openai \
  --api-base http://192.168.1.166:8000/v1 \
  --model gpt-4 \
  --context-size 512 \
  --rag-db /tmp/eviction_test_micro.db
```

---

## Test 1: Basic Eviction Trigger (TINY: 2048 tokens)

**Objective:** Verify eviction triggers when context fills up

### Steps:

1. Start Shepherd with 2048 token context (see setup above)

2. Watch the logs in another terminal:
   ```bash
   # If using systemd/journal
   journalctl -u shepherd -f | grep -i evict

   # Or if running in foreground
   # Stderr will show: [INFO] Evicting messages...
   ```

3. Send many medium-sized messages (each ~200 tokens):
   ```
   > Tell me a detailed story about Paris. Make it at least 500 words.
   > Tell me about the history of Python programming. At least 500 words.
   > Explain machine learning in detail. At least 500 words.
   > Describe neural networks comprehensively. At least 500 words.
   > Tell me about database systems in depth. At least 500 words.
   > Explain HTTP protocol thoroughly. At least 500 words.
   > Describe cloud computing in detail. At least 500 words.
   > Tell me about containerization. At least 500 words.
   > Explain microservices architecture. At least 500 words.
   > Describe REST APIs in depth. At least 500 words.
   ```

4. **Watch for eviction logs:**
   ```
   [INFO] Context utilization: 95%
   [INFO] Evicting messages [1, 2] to free space
   [INFO] Archived 2 messages to RAG
   ```

5. **Verify RAG database:**
   ```bash
   sqlite3 /tmp/eviction_test_tiny.db \
     "SELECT COUNT(*) FROM conversations;"

   # Should show > 0 conversations archived
   ```

### Expected Results:
- ✅ Eviction occurs after ~10 messages (context fills up)
- ✅ Logs show "Evicting messages"
- ✅ RAG database has archived conversations
- ✅ Shepherd continues working (doesn't crash)
- ✅ NO server errors (400) in logs

### PASS Criteria:
- At least 1 eviction event in logs
- At least 1 conversation in RAG database
- No "Context limit exceeded in server mode" errors

---

## Test 2: RAG Archival Verification (SMALL: 8192 tokens)

**Objective:** Verify evicted messages are correctly archived to RAG

### Steps:

1. Start Shepherd with 8192 token context

2. Send distinctive messages that you'll search for later:
   ```
   > MARKER_1: The capital of France is Paris and it has the Eiffel Tower.
   > MARKER_2: Python was created by Guido van Rossum in 1991.
   > MARKER_3: Machine learning is a subset of artificial intelligence.
   ```

3. Send many filler messages to force eviction of the markers:
   ```
   > Tell me about [topic]. Make it at least 1000 words.
   (Repeat with 20+ different topics)
   ```

4. **Check RAG database for archived markers:**
   ```bash
   sqlite3 /tmp/eviction_test_small.db \
     "SELECT user_message FROM conversations WHERE user_message LIKE '%MARKER_%' LIMIT 10;"

   # Should show the distinctive messages
   ```

5. **Search using SQL:**
   ```bash
   sqlite3 /tmp/eviction_test_small.db \
     "SELECT user_message, assistant_response FROM conversations WHERE user_message LIKE '%France%';"
   ```

### Expected Results:
- ✅ MARKER_1, MARKER_2, MARKER_3 found in RAG database
- ✅ Full user_message and assistant_response stored
- ✅ Timestamps present

### PASS Criteria:
- All 3 marker messages found in RAG
- Messages have corresponding responses
- Database schema is correct

---

## Test 3: RAG Retrieval with search_memory (SMALL: 8192 tokens)

**Objective:** Verify search_memory tool retrieves archived messages

### Steps:

1. Start Shepherd with 8192 token context

2. Send a distinctive message:
   ```
   > The quick brown fox jumps over the lazy dog near the riverbank.
   ```

3. Fill context to force eviction (20+ messages of 500+ words each)

4. **Use search_memory tool:**
   ```
   > search_memory('quick brown fox')
   ```

5. **Verify response mentions the archived message:**
   ```
   Response should contain: "I found this in my memory: ...quick brown fox..."
   ```

### Expected Results:
- ✅ search_memory finds the evicted message
- ✅ Returns relevant context
- ✅ Tool execution works correctly

### PASS Criteria:
- search_memory returns results
- Results contain "quick brown fox"
- Tool doesn't error

---

## Test 4: Server Never Sees Overflow (TINY: 2048 tokens)

**Objective:** Verify client manages context, server never gets errors

### Setup:
- **Client context:** 2048 tokens
- **Server context:** 98,304 tokens
- **Expected:** Client evicts before server sees overflow

### Steps:

1. Start Shepherd client with 2048 token context

2. Monitor **server logs** in another terminal:
   ```bash
   # On server machine
   journalctl -u shepherd -f | grep -i "error\|400"
   ```

3. Send 50+ messages to fill past client's 2048 limit:
   ```
   > [Send many 500-word messages]
   ```

4. **Watch client logs:**
   ```
   [INFO] Evicting messages  ← Should see this
   ```

5. **Watch server logs:**
   ```
   Should see NO errors like:
   [ERROR] Context limit exceeded in server mode
   ```

### Expected Results:
- ✅ Client logs show evictions
- ✅ Server logs show NO 400 errors
- ✅ Server only sees ≤ 2048 tokens at a time
- ✅ Conversation continues smoothly

### PASS Criteria:
- Client evicted at least 5 times
- Server logs have ZERO "Context limit exceeded" errors
- No 400 HTTP responses

---

## Test 5: Rapid Eviction (MICRO: 512 tokens)

**Objective:** Verify stable behavior with frequent evictions

### Steps:

1. Start Shepherd with 512 token context (very small!)

2. Send 100 small messages:
   ```
   > Message 1: [200 words]
   > Message 2: [200 words]
   ...
   > Message 100: [200 words]
   ```

3. **Count evictions:**
   ```bash
   # In logs, count "Evicting messages" lines
   grep -c "Evicting messages" /path/to/shepherd.log

   # Should be 50+ with 512-token context
   ```

4. **Check RAG:**
   ```bash
   sqlite3 /tmp/eviction_test_micro.db \
     "SELECT COUNT(*) FROM conversations;"

   # Should have many archived conversations
   ```

### Expected Results:
- ✅ 50+ eviction events
- ✅ 50+ conversations in RAG
- ✅ No crashes or errors
- ✅ Performance doesn't degrade

### PASS Criteria:
- Evictions ≥ 50
- RAG conversations ≥ 50
- Shepherd still responsive
- No memory leaks (check with `top`)

---

## Test 6: Conversation Coherence After Eviction

**Objective:** Verify conversation remains coherent after eviction

### Steps:

1. Start Shepherd with 2048 token context

2. Have a conversation about a specific topic:
   ```
   > Let's discuss Paris. What's special about it?
   (Get response)

   > Tell me more about the Eiffel Tower.
   (Get response)

   > What about the Louvre museum?
   (Get response)
   ```

3. Fill context to force eviction (send 20+ long messages on other topics)

4. **Reference the original topic:**
   ```
   > Earlier we discussed Paris. What did we talk about regarding the Eiffel Tower?
   ```

5. **Expected behavior:**
   - IF message still in context: Direct answer
   - IF message evicted to RAG: Should use search_memory tool to retrieve it

### Expected Results:
- ✅ Can reference recent context directly
- ✅ Uses search_memory for evicted context
- ✅ Conversation flows naturally
- ✅ No confusion or hallucination

### PASS Criteria:
- References to recent context work
- Optionally uses search_memory for old context
- Coherent responses

---

## Test 7: Edge Case - Protected Messages Only

**Objective:** Verify handling when only system+user messages exist

### Steps:

1. Start Shepherd with 512 token context

2. Configure large system prompt:
   ```bash
   # Edit ~/.shepherd/config.json
   {
     "system_prompt": "[4000-character system prompt...]"
   }
   ```

3. Send a large user message that exceeds remaining space:
   ```
   > [Send 400-token message when only 100 tokens remain]
   ```

4. **Expected:** Error or graceful handling (can't evict system/current user)

### Expected Results:
- ✅ Either accepts within available space
- ✅ Or returns error about context limit
- ✅ Does NOT crash
- ✅ Does NOT evict system prompt

### PASS Criteria:
- Shepherd handles gracefully
- System prompt never evicted
- Clear error message if applicable

---

## Test Results Summary Sheet

| Test | Description | Context Size | PASS/FAIL | Notes |
|------|-------------|--------------|-----------|-------|
| 1 | Basic Eviction | 2048 | | Evictions: ___ |
| 2 | RAG Archival | 8192 | | Markers found: ___/3 |
| 3 | search_memory | 8192 | | Tool worked: Y/N |
| 4 | Server No Errors | 2048 | | Server errors: ___ |
| 5 | Rapid Eviction | 512 | | Evictions: ___ |
| 6 | Conversation Coherence | 2048 | | Coherent: Y/N |
| 7 | Protected Messages | 512 | | Handled: Y/N |

---

## Quick Verification Commands

```bash
# Count evictions in logs
grep -c "Evicting messages" /path/to/shepherd.log

# Count RAG conversations
sqlite3 /tmp/eviction_test_tiny.db "SELECT COUNT(*) FROM conversations;"

# View recent archived conversations
sqlite3 /tmp/eviction_test_tiny.db \
  "SELECT timestamp, substr(user_message, 1, 50) FROM conversations ORDER BY timestamp DESC LIMIT 10;"

# Search RAG for specific content
sqlite3 /tmp/eviction_test_tiny.db \
  "SELECT user_message FROM conversations WHERE user_message LIKE '%search_term%';"

# Check database size
ls -lh /tmp/eviction_test_*.db
```

---

## Success Criteria (Overall)

For Shepherd CLIENT eviction to be considered **fully functional**:

- ✅ **All 7 tests PASS**
- ✅ **No server-side errors** when client context < server context
- ✅ **RAG archival works** (messages stored correctly)
- ✅ **RAG retrieval works** (search_memory finds messages)
- ✅ **Stable under load** (100+ evictions without crashes)
- ✅ **Conversation coherent** after eviction
- ✅ **System prompts protected** (never evicted)

---

## Troubleshooting

### No evictions occurring:
- Check context size is small enough (try 512 or 1024)
- Verify messages are long enough to fill context
- Check logs for actual token counts

### RAG database empty:
- Verify `--rag-db` path is writable
- Check shepherd has permissions
- Look for RAG-related errors in logs

### search_memory not working:
- Verify FTS5 index exists: `sqlite3 db "SELECT * FROM sqlite_master WHERE type='table';"`
- Check if conversations table has data
- Try direct SQL query first

### Server errors appearing:
- Client context size may be > server context
- Check both context sizes match expectations
- Verify api-base URL is correct

---

## Automated Test (Future)

To automate this testing, Shepherd would need:
1. **Batch mode:** `--batch` flag that processes stdin without prompts
2. **JSON output:** `--json` for machine-readable responses
3. **Test hooks:** Programmatic access to eviction events

Until then, manual testing per this plan is recommended.
