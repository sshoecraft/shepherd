# Shepherd Test Plan

## Overview

This document outlines the testing strategy for Shepherd, covering both automated tests and manual testing procedures. The goal is to ensure all features work correctly and regressions are caught early.

## Testing Philosophy

Given Shepherd's architecture, testing falls into several categories:
1. **Unit Tests** - Individual functions and classes (fully automatable)
2. **Integration Tests** - Component interactions (largely automatable)
3. **Backend Tests** - Provider/backend connectivity (automatable with mocking)
4. **API/Server Tests** - HTTP endpoints (fully automatable)
5. **CLI Tests** - Command-line interface (partially automatable)
6. **TUI Tests** - Visual interface (requires manual testing)
7. **End-to-End Tests** - Full workflows (mix of automated and manual)

---

## Part 1: Automated Testing

### 1.1 Unit Tests

#### Config Module (`config.cpp/h`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CFG-001 | Load valid config file | Config object populated correctly |
| CFG-002 | Load missing config file | Default values applied |
| CFG-003 | Parse size strings ("10G", "500M") | Correct byte values |
| CFG-004 | Save config to disk | JSON file created with correct structure |
| CFG-005 | `handle_config_args(["show"])` | Returns 0, prints config |
| CFG-006 | `handle_config_args(["set", "streaming", "true"])` | Updates config, saves |
| CFG-007 | Invalid config key | Error reported, returns non-zero |

#### Provider Module (`provider.cpp/h`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| PRV-001 | Load providers from directory | All JSON files parsed |
| PRV-002 | `handle_provider_args(["list"])` | Lists all providers |
| PRV-003 | `handle_provider_args(["add", ...])` | Creates provider file |
| PRV-004 | `handle_provider_args(["show", "name"])` | Shows provider details |
| PRV-005 | `handle_provider_args(["use", "name"])` | Sets current provider |
| PRV-006 | `handle_provider_args(["next"])` | Cycles to next provider |
| PRV-007 | Invalid provider name | Error message, non-zero return |
| PRV-008 | OAuth token refresh | Token acquired, cached, refreshed |

#### Session Module (`session.cpp/h`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| SES-001 | Add user message | Message stored with token count |
| SES-002 | Add system message | Stored at index 0 |
| SES-003 | Add tool response | Correct role and tool_id |
| SES-004 | Token counting | Accurate per-message counts |
| SES-005 | Context eviction (turn-based) | Oldest turns removed first |
| SES-006 | Context eviction (message-based) | Individual messages removed |
| SES-007 | System message protection | Never evicted |
| SES-008 | Clear history | Only system message remains |
| SES-009 | Serialize to JSON | Valid JSON with all messages |

#### Scheduler Module (`scheduler.cpp/h`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| SCH-001 | Parse cron expression `* * * * *` | Matches every minute |
| SCH-002 | Parse cron expression `0 9 * * 1` | Monday 9:00 AM |
| SCH-003 | Parse cron range `10-20` | Values 10-20 inclusive |
| SCH-004 | Parse cron step `*/5` | Values 0,5,10,15... |
| SCH-005 | `handle_sched_args(["add", ...])` | Creates schedule entry |
| SCH-006 | `handle_sched_args(["remove", "name"])` | Removes schedule |
| SCH-007 | `handle_sched_args(["enable", "name"])` | Sets enabled=true |
| SCH-008 | `handle_sched_args(["disable", "name"])` | Sets enabled=false |
| SCH-009 | Schedule persistence | Saved to JSON, reloaded |
| SCH-010 | `next` calculation | Correct next execution time |

#### Tools Module (`tools/*.cpp`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TL-001 | Register tool | Tool accessible by name |
| TL-002 | Execute valid tool | Result returned |
| TL-003 | Execute disabled tool | Error returned |
| TL-004 | Tool enable/disable | State persisted |
| TL-005 | `handle_tools_args(["list"])` | Lists all tools |
| TL-006 | `handle_tools_args(["enable", "tool"])` | Tool enabled |
| TL-007 | `handle_tools_args(["disable", "tool"])` | Tool disabled |

##### Filesystem Tools

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TL-FS-001 | `read_file` existing file | Content returned |
| TL-FS-002 | `read_file` missing file | Error with message |
| TL-FS-003 | `write_file` new file | File created |
| TL-FS-004 | `write_file` overwrite | Content replaced |
| TL-FS-005 | `list_directory` valid path | Files listed |
| TL-FS-006 | `list_directory` invalid path | Error returned |
| TL-FS-007 | `delete_file` existing | File removed |
| TL-FS-008 | `file_exists` check | Boolean result |

##### Command Tools

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TL-CMD-001 | `execute_command` simple | Output returned |
| TL-CMD-002 | `execute_command` with timeout | Times out correctly |
| TL-CMD-003 | `execute_command` exit code | Non-zero exit reported |
| TL-CMD-004 | `get_environment_variable` | Value returned |

##### HTTP Tools

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TL-HTTP-001 | GET request | Response body returned |
| TL-HTTP-002 | POST request | Body sent, response returned |
| TL-HTTP-003 | Custom headers | Headers included |
| TL-HTTP-004 | Error status codes | Error reported |
| TL-HTTP-005 | Timeout handling | Timeout error returned |

##### Memory Tools

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TL-MEM-001 | `set_fact` | Fact stored |
| TL-MEM-002 | `get_fact` existing | Value returned |
| TL-MEM-003 | `get_fact` missing | Not found error |
| TL-MEM-004 | `clear_fact` | Fact removed |
| TL-MEM-005 | `store_memory` | Q&A pair stored |
| TL-MEM-006 | `search_memory` | Relevant results returned |
| TL-MEM-007 | `clear_memory` | Memory cleared |

##### JSON Tools

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TL-JSON-001 | Parse valid JSON | Parsed object returned |
| TL-JSON-002 | Parse invalid JSON | Error message |
| TL-JSON-003 | Serialize object | Valid JSON string |
| TL-JSON-004 | Query with JSONPath | Extracted values |

#### Models Module (`backends/models.cpp`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| MDL-001 | Detect Qwen from template | `QWEN_2_X` or `QWEN_3_X` |
| MDL-002 | Detect Llama3 from template | `LLAMA_3_X` |
| MDL-003 | Detect GLM4 from template | `GLM_4` |
| MDL-004 | Detect from config.json | Correct family |
| MDL-005 | Detect from model path | Correct family |
| MDL-006 | API model lookup | Correct context size |

#### ChatTemplate Module (`backends/chat_template.cpp`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| TPL-001 | ChatML format_message | Correct `<\|im_start\|>...<\|im_end\|>` |
| TPL-002 | Llama3 format_message | Correct header tags |
| TPL-003 | System message with tools | Tools embedded correctly |
| TPL-004 | Generation prompt | Correct assistant start |
| TPL-005 | Minja template rendering | Correct output |
| TPL-006 | Incremental rendering | Only new content returned |
| TPL-007 | Capability probing | Correct caps detected |

### 1.2 Integration Tests

#### Backend Integration

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| BK-001 | LlamaCpp load model | Model loads, context created |
| BK-002 | LlamaCpp tokenization | Correct token counts |
| BK-003 | LlamaCpp generation | Response generated |
| BK-004 | LlamaCpp KV cache | Tokens cached correctly |
| BK-005 | LlamaCpp eviction | Old messages evicted |
| BK-006 | OpenAI API request | Valid response |
| BK-007 | Anthropic API request | Valid response |
| BK-008 | Gemini API request | Valid response |
| BK-009 | Ollama API request | Valid response |
| BK-010 | Streaming callback | Deltas received |
| BK-011 | Tool call detection | Tool calls parsed |
| BK-012 | Cancellation | Generation stops |

#### Frontend Integration

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| FE-001 | CLI init with provider | Backend connected |
| FE-002 | CLI tool initialization | All tools registered |
| FE-003 | GenerationThread start | Thread running |
| FE-004 | GenerationThread submit | Request processed |
| FE-005 | EventCallback invocation | Events received |

### 1.3 Server Tests

#### API Server (`/v1/chat/completions`)

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| API-001 | POST valid request | 200 with completion |
| API-002 | POST with streaming | SSE events received |
| API-003 | POST with tools | Tool calls in response |
| API-004 | POST invalid JSON | 400 error |
| API-005 | GET /v1/models | Model list returned |
| API-006 | GET /health | Healthy status |
| API-007 | Tool result submission | Processed correctly |
| API-008 | Context limit handling | Appropriate error |
| API-009 | Prefix caching | Fast subsequent requests |

#### CLI Server

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CLIS-001 | POST /request | Response generated |
| CLIS-002 | GET /updates SSE | Events streamed |
| CLIS-003 | POST /clear | Session cleared |
| CLIS-004 | GET /session | Session state returned |
| CLIS-005 | Tool execution | Tool runs server-side |
| CLIS-006 | SSE broadcast | All clients receive |

### 1.4 CLI Command Tests

| Test ID | Test Case | Expected Result |
|---------|-----------|-----------------|
| CLI-001 | `shepherd --help` | Help text displayed |
| CLI-002 | `shepherd --version` | Version displayed |
| CLI-003 | `shepherd config show` | Config displayed |
| CLI-004 | `shepherd config set key value` | Config updated |
| CLI-005 | `shepherd provider list` | Providers listed |
| CLI-006 | `shepherd provider add ...` | Provider created |
| CLI-007 | `shepherd provider use name` | Provider selected |
| CLI-008 | `shepherd sched list` | Schedules listed |
| CLI-009 | `shepherd tools list` | Tools listed |
| CLI-010 | `shepherd mcp list` | MCP servers listed |

---

## Part 2: Manual Testing

### 2.1 CLI Interactive Mode

#### Basic Interaction

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| CLI-M-001 | Start shepherd, type "Hello" | Response generated | |
| CLI-M-002 | Type multiline input (paste) | Input accepted, response generated | |
| CLI-M-003 | Press Up arrow | Previous input recalled | |
| CLI-M-004 | Press Down arrow | Next input recalled | |
| CLI-M-005 | Press Ctrl+A | Cursor moves to start | |
| CLI-M-006 | Press Ctrl+E | Cursor moves to end | |
| CLI-M-007 | Press Ctrl+U | Line cleared | |
| CLI-M-008 | Press Ctrl+K | Text after cursor deleted | |
| CLI-M-009 | Press Escape during generation | Generation cancelled | |
| CLI-M-010 | Press Ctrl+D on empty line | Exit gracefully | |

#### Slash Commands

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| CLI-M-011 | Type `/provider list` | Providers listed | |
| CLI-M-012 | Type `/provider use <name>` | Provider switched | |
| CLI-M-013 | Type `/provider next` | Next provider selected | |
| CLI-M-014 | Type `/config show` | Config displayed | |
| CLI-M-015 | Type `/config set streaming true` | Config updated | |
| CLI-M-016 | Type `/tools list` | Tools listed | |
| CLI-M-017 | Type `/tools enable <tool>` | Tool enabled | |
| CLI-M-018 | Type `/tools disable <tool>` | Tool disabled | |
| CLI-M-019 | Type `/sched list` | Schedules listed | |
| CLI-M-020 | Type `exit` or `quit` | Exit gracefully | |

#### Tool Execution

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| CLI-M-021 | Ask to read a file | File content displayed | |
| CLI-M-022 | Ask to write a file | File created | |
| CLI-M-023 | Ask to run a command | Output displayed | |
| CLI-M-024 | Ask to make HTTP request | Response returned | |
| CLI-M-025 | Ask to store a fact | Fact stored | |
| CLI-M-026 | Ask to recall a fact | Fact retrieved | |
| CLI-M-027 | Ask to search memory | Results returned | |
| CLI-M-028 | Trigger multiple tool calls | All tools executed | |

#### Color Output

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| CLI-M-029 | Normal response | Default terminal color | |
| CLI-M-030 | User input echo | Green text | |
| CLI-M-031 | Tool call display | Yellow text | |
| CLI-M-032 | Tool result display | Cyan text | |
| CLI-M-033 | Error message | Red text | |
| CLI-M-034 | Set NO_COLOR=1 | No colors | |

### 2.2 TUI Mode

#### Window Layout

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| TUI-001 | Start with `--tui` | TUI displayed correctly | |
| TUI-002 | Check output window | Scrollable content area | |
| TUI-003 | Check input box | Bordered input area | |
| TUI-004 | Check status line | Model info, token counts | |
| TUI-005 | Resize terminal | Layout adapts correctly | |

#### Input Handling

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| TUI-006 | Type text | Characters appear in input box | |
| TUI-007 | Press Enter | Input submitted, appears in output | |
| TUI-008 | Use arrow keys | Cursor moves in input box | |
| TUI-009 | Use history (Up/Down) | Previous inputs recalled | |
| TUI-010 | Paste multiline text | Input box expands | |

#### Output Handling

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| TUI-011 | Generate response | Text streams to output window | |
| TUI-012 | Page Up/Down | Scrolls output window | |
| TUI-013 | Auto-scroll | Scrolls to bottom on new content | |
| TUI-014 | Tool call display | Formatted tool call shown | |
| TUI-015 | Thinking block (--thinking) | Thinking displayed in gray | |

#### Status Updates

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| TUI-016 | Check token count | Updates after each message | |
| TUI-017 | Switch provider | Status shows new model | |
| TUI-018 | During generation | Shows generating state | |

#### Queue Display

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| TUI-019 | Queue input during generation | Input shown in pending area | |
| TUI-020 | Multiple queued inputs | All shown stacked | |
| TUI-021 | Processing queued input | Current shows as cyan | |

### 2.3 Server Mode Testing

#### API Server Mode

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| SRV-001 | Start with `--apiserver` | Server starts, logs endpoints | |
| SRV-002 | curl /health | `{"status": "healthy"}` | |
| SRV-003 | curl /v1/models | Model list returned | |
| SRV-004 | POST /v1/chat/completions | Response generated | |
| SRV-005 | POST with stream:true | SSE events received | |
| SRV-006 | Use OpenAI Python client | Compatible responses | |
| SRV-007 | Control socket status | Status returned | |
| SRV-008 | Control socket shutdown | Server stops gracefully | |

#### CLI Server Mode

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| SRV-009 | Start with `--cliserver` | Server starts, logs endpoints | |
| SRV-010 | POST /request | Response generated | |
| SRV-011 | GET /updates SSE | Events stream to client | |
| SRV-012 | GET /session | Session state returned | |
| SRV-013 | POST /clear | Session cleared | |
| SRV-014 | Multiple clients | All receive SSE events | |
| SRV-015 | Tool execution | Tools run server-side | |

### 2.4 Multi-Provider Testing

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| MPV-001 | Configure multiple providers | All saved correctly | |
| MPV-002 | Switch providers mid-session | Context maintained where possible | |
| MPV-003 | Provider fallback on failure | Next provider tried | |
| MPV-004 | API tools (ask_*) | Other providers available as tools | |
| MPV-005 | Cross-model consultation | ask_* tools work | |

### 2.5 Backend-Specific Testing

#### LlamaCpp Backend

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| LC-001 | Load GGUF model | Model loads successfully | |
| LC-002 | Multi-GPU (if available) | Uses configured GPUs | |
| LC-003 | KV cache eviction | Older messages evicted | |
| LC-004 | Long conversation | Handles eviction smoothly | |
| LC-005 | Token limit reached | Appropriate handling | |

#### TensorRT Backend

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| TRT-001 | Load TRT-LLM model | Model loads successfully | |
| TRT-002 | Multi-GPU inference | Distributed correctly | |
| TRT-003 | KV cache events | Events processed | |

#### API Backends

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| APIB-001 | OpenAI connection | Responses generated | |
| APIB-002 | Anthropic connection | Responses generated | |
| APIB-003 | Gemini connection | Responses generated | |
| APIB-004 | Ollama connection | Responses generated | |
| APIB-005 | OAuth authentication | Token acquired and used | |
| APIB-006 | Azure OpenAI | Responses generated | |
| APIB-007 | Rate limit handling | Appropriate retry/error | |

### 2.6 Edge Cases

| Test ID | Procedure | Expected Result | Pass/Fail |
|---------|-----------|-----------------|-----------|
| EDGE-001 | Very long input | Handles gracefully | |
| EDGE-002 | Empty input | Ignored or appropriate error | |
| EDGE-003 | Special characters in input | Handled correctly | |
| EDGE-004 | Binary file read attempt | Appropriate error | |
| EDGE-005 | Network disconnection | Timeout and error | |
| EDGE-006 | Invalid API key | Clear error message | |
| EDGE-007 | Full disk (write file) | Appropriate error | |
| EDGE-008 | Permission denied | Clear error message | |
| EDGE-009 | Rapid input submission | Queue handled correctly | |
| EDGE-010 | Concurrent requests (server) | Serialized correctly | |

---

## Part 3: Test Infrastructure

### 3.1 Recommended Test Framework

For C++ testing, recommend:
- **Google Test (gtest)** - For unit tests
- **Google Mock** - For mocking backends/HTTP
- **Catch2** - Alternative lightweight framework

### 3.2 Test Directory Structure

```
tests/
├── unit/
│   ├── test_config.cpp
│   ├── test_provider.cpp
│   ├── test_session.cpp
│   ├── test_scheduler.cpp
│   ├── test_tools.cpp
│   ├── test_models.cpp
│   └── test_chat_template.cpp
├── integration/
│   ├── test_backend_llamacpp.cpp
│   ├── test_backend_openai.cpp
│   ├── test_backend_anthropic.cpp
│   ├── test_frontend_cli.cpp
│   └── test_server.cpp
├── fixtures/
│   ├── models/           # Test model configs
│   ├── providers/        # Test provider configs
│   └── templates/        # Test chat templates
├── mocks/
│   ├── mock_backend.h
│   ├── mock_http_client.h
│   └── mock_tool.h
└── scripts/
    ├── run_unit_tests.sh
    ├── run_integration_tests.sh
    ├── run_server_tests.sh
    └── run_all_tests.sh
```

### 3.3 Mock Strategies

#### HTTP Mock
```cpp
class MockHttpClient : public HttpClient {
    std::map<std::string, std::string> responses;
public:
    void set_response(const std::string& url, const std::string& response);
    Response get(const std::string& url) override;
    Response post(const std::string& url, const std::string& body) override;
};
```

#### Backend Mock
```cpp
class MockBackend : public Backend {
    std::string canned_response;
public:
    void set_response(const std::string& response);
    Response add_message(Session& session, ...) override;
};
```

### 3.4 CI/CD Integration

#### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: make test-build
      - name: Run Unit Tests
        run: make test-unit

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Run Integration Tests
        run: make test-integration

  server-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Run Server Tests
        run: make test-server
```

### 3.5 Test Coverage Goals

| Component | Target Coverage |
|-----------|-----------------|
| Config | 90% |
| Provider | 90% |
| Session | 85% |
| Scheduler | 90% |
| Tools | 80% |
| Models | 85% |
| ChatTemplate | 80% |
| Backends | 70% |
| Frontend | 60% |
| Server | 75% |

---

## Part 4: Test Execution Checklist

### Pre-Release Testing Checklist

#### Build Verification
- [ ] Clean build succeeds
- [ ] All compiler warnings addressed
- [ ] Debug build works
- [ ] Release build works

#### Unit Tests
- [ ] All unit tests pass
- [ ] Coverage targets met

#### Integration Tests
- [ ] All integration tests pass
- [ ] Backend connections verified

#### Manual Testing
- [ ] CLI interactive mode verified
- [ ] TUI mode verified
- [ ] Server modes verified
- [ ] All slash commands work
- [ ] Tool execution verified

#### Platform Testing
- [ ] Linux verified
- [ ] macOS verified (if applicable)
- [ ] Multi-GPU verified (if applicable)

#### Documentation
- [ ] CHANGELOG updated
- [ ] Version number incremented
- [ ] README accurate

---

## Appendix A: Test Data

### Sample Provider Config
```json
{
  "name": "test-openai",
  "type": "openai",
  "model": "gpt-4",
  "api_key": "sk-test-key",
  "base_url": "https://api.openai.com/v1"
}
```

### Sample Schedule
```json
{
  "id": "test123",
  "name": "test-schedule",
  "cron": "0 9 * * 1",
  "prompt": "Test prompt",
  "enabled": true
}
```

### Sample Tool Call JSON
```json
{
  "name": "read_file",
  "arguments": {
    "path": "/tmp/test.txt"
  }
}
```

---

## Appendix B: Troubleshooting Tests

### Common Test Failures

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Config test fails | Missing test fixtures | Create fixture files |
| Backend test timeout | Network issues | Check connectivity, use mocks |
| Server test port conflict | Port in use | Use dynamic port allocation |
| TUI test visual mismatch | Terminal differences | Document expected behavior |
| Tool test permission error | File permissions | Run with appropriate permissions |

### Debug Flags

```bash
# Enable debug output
SHEPHERD_DEBUG=3 ./shepherd

# Enable specific backend debug
LLAMA_LOG_LEVEL=debug ./shepherd

# Test with verbose HTTP
SHEPHERD_HTTP_DEBUG=1 ./shepherd
```
