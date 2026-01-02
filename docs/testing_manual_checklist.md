# Shepherd Manual Testing Checklist

This checklist is for manual testing that cannot be easily automated. Check each item as you test.

## Quick Test Protocol

For rapid regression testing before releases:
1. Run Quick Smoke Test (5 minutes)
2. Run Core Feature Tests (15 minutes)
3. Run Backend Tests for configured providers (10 minutes per backend)

---

## Quick Smoke Test (~5 minutes)

### Build & Launch
- [ ] `make clean && make` succeeds without errors
- [ ] `./shepherd --version` shows current version
- [ ] `./shepherd --help` displays help text

### Basic Interaction
- [ ] Launch shepherd with default provider
- [ ] Type "Hello, how are you?" - response generated
- [ ] Type `/provider list` - providers listed
- [ ] Type `exit` - exits cleanly

---

## Core Feature Tests (~15 minutes)

### CLI Input Handling

| Test | Action | Expected | Pass |
|------|--------|----------|------|
| Basic input | Type "What is 2+2?" | Response with "4" | [ ] |
| History up | Press Up arrow | Previous input shown | [ ] |
| History down | Press Down arrow | Next/empty input | [ ] |
| Cursor home | Press Ctrl+A | Cursor at start | [ ] |
| Cursor end | Press Ctrl+E | Cursor at end | [ ] |
| Clear line | Press Ctrl+U | Line cleared | [ ] |
| Cancel | Press Escape during generation | Generation stops | [ ] |
| Exit | Press Ctrl+D on empty line | Clean exit | [ ] |

### Slash Commands

| Command | Expected Result | Pass |
|---------|-----------------|------|
| `/provider list` | Shows all providers | [ ] |
| `/provider show <name>` | Shows provider details | [ ] |
| `/provider use <name>` | Switches provider | [ ] |
| `/provider next` | Cycles to next provider | [ ] |
| `/config show` | Shows configuration | [ ] |
| `/config set streaming true` | Updates streaming | [ ] |
| `/config set streaming false` | Updates streaming | [ ] |
| `/tools list` | Lists all tools | [ ] |
| `/sched list` | Lists schedules | [ ] |

### Tool Execution

Ask the model to perform these tasks and verify:

| Task | Command to Model | Verify | Pass |
|------|------------------|--------|------|
| Read file | "Read /etc/hostname" | Content displayed | [ ] |
| Write file | "Write 'test' to /tmp/test.txt" | File created | [ ] |
| List directory | "List files in /tmp" | Files shown | [ ] |
| Run command | "Run: echo hello" | "hello" output | [ ] |
| Store fact | "Remember my name is TestUser" | Confirmation | [ ] |
| Recall fact | "What is my name?" | "TestUser" | [ ] |
| Get date | "What is today's date?" | Current date | [ ] |

### Multi-Turn Conversation

1. [ ] Start conversation: "My favorite color is blue"
2. [ ] Follow up: "What did I just tell you?"
3. [ ] Verify: Model recalls "blue" or "favorite color"

### Error Handling

| Scenario | Action | Expected | Pass |
|----------|--------|----------|------|
| Invalid provider | `/provider use nonexistent` | Error message | [ ] |
| Invalid tool | Trigger non-existent tool | Error handled | [ ] |
| Network error | Disconnect network, send message | Timeout/error | [ ] |

---

## TUI Mode Tests (~10 minutes)

Start with: `./shepherd --tui`

### Layout

| Check | What to Verify | Pass |
|-------|----------------|------|
| Output window | Main area shows conversation | [ ] |
| Input box | Bordered area at bottom | [ ] |
| Status line | Shows model name + tokens | [ ] |
| Resize | Resize terminal - layout adapts | [ ] |

### Input

| Test | Action | Expected | Pass |
|------|--------|----------|------|
| Typing | Type characters | Appear in input box | [ ] |
| Submit | Press Enter | Input sent, shown in output | [ ] |
| History | Up/Down arrows | Cycles through history | [ ] |
| Multiline | Paste multiple lines | Input box expands | [ ] |

### Output

| Test | Action | Expected | Pass |
|------|--------|----------|------|
| Streaming | Send message | Text streams in output | [ ] |
| Scroll | Page Up/Down | Output scrolls | [ ] |
| Auto-scroll | New content arrives | Scrolls to bottom | [ ] |
| Colors | Various message types | Different colors shown | [ ] |

### Status

| Check | Verify | Pass |
|-------|--------|------|
| Token count | Updates after messages | [ ] |
| Model name | Shows current model | [ ] |
| Provider change | `/provider use X` updates status | [ ] |

---

## Server Mode Tests (~15 minutes)

### API Server Mode

Start: `./shepherd --apiserver --port 8080`

| Test | Command | Expected | Pass |
|------|---------|----------|------|
| Health check | `curl localhost:8080/health` | `{"status":"healthy"}` | [ ] |
| Models list | `curl localhost:8080/v1/models` | Model list JSON | [ ] |
| Chat completion | See below | Response JSON | [ ] |
| Streaming | See below | SSE events | [ ] |
| Control status | `shepherd ctl status` | Status JSON | [ ] |
| Shutdown | `shepherd ctl shutdown` | Server stops | [ ] |

Chat completion test:
```bash
curl -X POST localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}]}'
```
- [ ] Valid response JSON returned

Streaming test:
```bash
curl -X POST localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```
- [ ] Multiple `data:` events received
- [ ] Final `data: [DONE]` received

### CLI Server Mode

Start: `./shepherd --cliserver --port 8081`

| Test | Command | Expected | Pass |
|------|---------|----------|------|
| Request | POST /request with prompt | Response | [ ] |
| Session | GET /session | Session state | [ ] |
| Updates | GET /updates | SSE stream | [ ] |
| Clear | POST /clear | Session cleared | [ ] |

---

## Backend-Specific Tests

### LlamaCpp Backend

Prerequisites: GGUF model file available

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Load model | Start with GGUF path | Model loads | [ ] |
| Generation | Send message | Response generated | [ ] |
| Token count | Check status | Tokens reported | [ ] |
| Long convo | Send 10+ messages | No errors | [ ] |
| Context fill | Fill context window | Eviction occurs | [ ] |

### OpenAI Backend

Prerequisites: Valid API key configured

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Connect | `/provider use openai` | Connected | [ ] |
| Chat | Send message | GPT response | [ ] |
| Streaming | Send longer prompt | Tokens stream | [ ] |
| Tools | Request tool use | Tool called | [ ] |

### Anthropic Backend

Prerequisites: Valid API key configured

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Connect | `/provider use anthropic` | Connected | [ ] |
| Chat | Send message | Claude response | [ ] |
| Streaming | Send longer prompt | Tokens stream | [ ] |
| Tools | Request tool use | Tool called | [ ] |

### Gemini Backend

Prerequisites: Valid API key configured

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Connect | `/provider use gemini` | Connected | [ ] |
| Chat | Send message | Gemini response | [ ] |
| Tools | Request tool use | Tool called | [ ] |

### Ollama Backend

Prerequisites: Ollama server running

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Connect | `/provider use ollama` | Connected | [ ] |
| Chat | Send message | Response | [ ] |
| Model list | `/model list` | Models shown | [ ] |

---

## Advanced Feature Tests

### Provider Switching

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Mid-conversation | Switch after 3 messages | New provider works | [ ] |
| Provider fallback | Use failing provider, then valid | Falls back | [ ] |
| API tools | Use `ask_<provider>` tool | Cross-model works | [ ] |

### Scheduler

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Add schedule | `/sched add test "* * * * *" "hi"` | Created | [ ] |
| List | `/sched list` | Shows schedule | [ ] |
| Disable | `/sched disable test` | Disabled | [ ] |
| Enable | `/sched enable test` | Enabled | [ ] |
| Remove | `/sched remove test` | Removed | [ ] |
| Execution | Wait for scheduled time | Prompt injected | [ ] |

### MCP Integration (if configured)

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| List MCP tools | `/tools list` | MCP tools shown | [ ] |
| Execute MCP tool | Trigger MCP tool | Tool executes | [ ] |

### Memory/RAG

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Store memory | Ask to remember info | Stored | [ ] |
| Search memory | Ask to recall | Found | [ ] |
| Fact storage | Set fact, new session, get fact | Retrieved | [ ] |
| Eviction archive | Fill context, check RAG | Archived | [ ] |

---

## Stress Tests (Optional)

| Test | Procedure | Expected | Pass |
|------|-----------|----------|------|
| Rapid input | Type 10 messages quickly | All processed | [ ] |
| Large file | Read 1MB file | Handled (maybe truncated) | [ ] |
| Long response | Request very long output | Streams correctly | [ ] |
| Many tools | Trigger 5+ tool calls | All execute | [ ] |

---

## Edge Case Tests

| Scenario | Test | Expected | Pass |
|----------|------|----------|------|
| Empty input | Press Enter with nothing | Ignored or error | [ ] |
| Unicode | Send emoji/unicode | Handled correctly | [ ] |
| Special chars | Send `<>&'"` | No injection issues | [ ] |
| Binary file | Try to read /bin/ls | Error or binary notice | [ ] |
| No permission | Read protected file | Permission error | [ ] |
| Network down | Send message offline | Timeout error | [ ] |
| Invalid JSON | Malformed tool call | Error handled | [ ] |

---

## Test Results Summary

| Category | Tests | Passed | Failed |
|----------|-------|--------|--------|
| Smoke Test | 4 | | |
| CLI Input | 8 | | |
| Slash Commands | 10 | | |
| Tool Execution | 7 | | |
| TUI Mode | 16 | | |
| API Server | 6 | | |
| CLI Server | 4 | | |
| Backend-Specific | varies | | |
| Advanced | varies | | |

**Tester:** _________________
**Date:** _________________
**Version Tested:** _________________
**Overall Result:** [ ] PASS  [ ] FAIL

**Notes:**
```




```
