# Parser Refactor Plan

## Current Problems

### 1. O(n²) Parser Performance
The current harmony parser (ported from llama.cpp) re-parses the **entire accumulated buffer** on every token:

```cpp
void Parser::feed(const std::string& text) {
    accumulated += text;
    accumulated_result = parse(accumulated, opts);  // Re-parse EVERYTHING
}
```

For 2048 tokens, this means ~2 million parse operations. Both llama.cpp and our port have this issue.

The official Harmony Rust library (`~/src/harmony`) has a proper O(n) state machine parser (`StreamableParser`), but we moved away from the Rust FFI approach.

### 2. Harmony Code Scattered Through Backend
52 harmony-specific conditionals scattered through `backends/llamacpp.cpp`:

```cpp
harmony_enabled = caps && caps->has_channels && !config->raw_output;
if (harmony_enabled) { ... }
if (harmony_enabled && harmony_parser && (eog_text == "<|return|>" ...)) { ... }
bool is_harmony_stop = false;
if (is_harmony_stop) { ... }
```

When tensorrt backend is added, all this would need to be duplicated.

### 3. The 2048 Token Issue (SEPARATE PROBLEM)
Shepherd generates 2048 tokens (hits max_tokens) while llama-server and vLLM stop naturally at ~1950 tokens on the same prompt.

- Shepherd: Never generates `<|return|>` within 2048 tokens
- llama-server: Generates `<|return|>` around 1950 tokens
- vLLM: Generates `<|return|>` around 1950 tokens

Same model, same sampling params (temp=1.0, top_p=1.0). The parser refactor **will not fix this** - it's a sampling/model behavior difference upstream of the parser.

---

## Proposed Architecture

### Abstract Parser Class

```cpp
// backends/parser.h (base class only)

class Parser {
public:
    // Process token text, returns true = STOP generation
    virtual bool process(const std::string& token) = 0;

    // Get deltas since last call
    virtual std::string get_content_delta() = 0;
    virtual std::string get_reasoning_delta() = 0;

    // Reset for new generation
    virtual void reset() = 0;

    virtual ~Parser() = default;
};
```

### GenericParser (Default)

For non-harmony models - absorbs `GpuBackend::output()` functionality (in `backends/generic_parser.h` and `backends/generic_parser.cpp`):

**Handles:**
- Code block tracking (backticks) - don't parse tags inside code blocks
- Think tag detection (`<think>`, `</think>`, etc.) - extracts to reasoning_delta
- Tool call detection (`<tool_call>`, `{...}`, etc.) - buffers for later parsing

```cpp
// backends/generic_parser.h
class GenericParser : public Parser {
public:
    bool process(const std::string& token) override;
    std::string get_content_delta() override;
    std::string get_reasoning_delta() override;
    void reset() override;

    // Configuration - set from backend's marker lists
    void set_thinking_markers(const std::vector<std::string>& start,
                              const std::vector<std::string>& end);
    void set_tool_markers(const std::vector<std::string>& start,
                          const std::vector<std::string>& end);

    // Get buffered tool call content (for parsing after generation)
    std::string get_tool_call_buffer() const { return buffered_tool_call; }
    bool has_tool_call() const { return !buffered_tool_call.empty(); }

private:
    // State machine (mirrors GpuBackend::output() FilterState)
    enum class State {
        NORMAL,
        DETECTING_TAG,
        IN_THINKING,
        IN_TOOL_CALL,
        CHECKING_CLOSE
    };

    void process_char(char c);
    bool matches_any(const std::string& buffer, const std::vector<std::string>& markers);
    bool could_match_any(const std::string& buffer, const std::vector<std::string>& markers);

    State state = State::NORMAL;
    bool in_code_block = false;
    int json_brace_depth = 0;

    std::string backtick_buffer;
    std::string tag_buffer;
    std::string pending_content;
    std::string pending_reasoning;
    std::string buffered_tool_call;

    // Configurable markers (set by backend)
    std::vector<std::string> thinking_start_markers;
    std::vector<std::string> thinking_end_markers;
    std::vector<std::string> tool_start_markers;
    std::vector<std::string> tool_end_markers;
};
```

**State Transitions (same as current GpuBackend::output()):**
```
NORMAL → saw '<' or '{' → DETECTING_TAG
DETECTING_TAG → matches thinking_start → IN_THINKING
DETECTING_TAG → matches tool_start or '{' → IN_TOOL_CALL
IN_THINKING → matches thinking_end → NORMAL
IN_TOOL_CALL → matches tool_end or balanced braces → NORMAL (emit tool call)
```

**Code Block Awareness:**
- Track triple backticks to toggle `in_code_block`
- When `in_code_block`, don't parse tags - treat `<tool_call>` as content

### HarmonyParser (GPT-OSS)

Character-by-character state machine for harmony format (in `backends/harmony_parser.h` and `backends/harmony_parser.cpp`):

```cpp
// backends/harmony_parser.h
class HarmonyParser : public Parser {
public:
    bool process(const std::string& token) override;
    std::string get_content_delta() override;
    std::string get_reasoning_delta() override;
    void reset() override;

private:
    // High-level parsing state
    enum class ParseState {
        EXPECT_START,   // Waiting for <|start|>
        HEADER,         // In header, waiting for <|message|>
        CONTENT         // In content, waiting for <|end|>, <|return|>, <|call|>
    };

    // Character-level marker detection state
    enum class CharState {
        NORMAL,           // Regular content
        SAW_LT,           // Saw '<'
        SAW_LT_PIPE,      // Saw '<|'
        MATCHING_MARKER   // Matching against known marker names
    };

    void process_char(char c);

    ParseState parse_state = ParseState::EXPECT_START;
    CharState char_state = CharState::NORMAL;
    std::string marker_buffer;        // Partial marker being matched
    std::string current_channel;      // "analysis", "final", etc.
    std::string pending_content;      // Content delta for final channel
    std::string pending_reasoning;    // Content delta for analysis channel
    bool should_stop = false;
};
```

**Harmony Format Markers:**
- `<|start|>` - Start of message
- `<|message|>` - End of header, start of content
- `<|channel|>` - Channel marker in header (followed by channel name)
- `<|end|>` - End of message (continue generating next message)
- `<|return|>` - Stop generation (final response complete)
- `<|call|>` - Tool call, stop generation

**State Transitions:**
```
EXPECT_START → saw <|start|> → HEADER
HEADER → saw <|message|> → CONTENT (extract channel from header)
CONTENT → saw <|end|> → EXPECT_START (message done, continue)
CONTENT → saw <|return|> → return true (STOP)
CONTENT → saw <|call|> → return true (STOP)
```

**Channel Handling:**
- `analysis` channel → `pending_reasoning` (thinking/reasoning)
- `final` channel → `pending_content` (user-facing response)
- Other channels → ignored or logged

---

## Backend Integration

### Backend Holds Parser Pointer

```cpp
// In Backend class
class Backend {
protected:
    std::unique_ptr<Parser> parser;

    void init_parser() {
        if (is_harmony_model()) {
            parser = std::make_unique<HarmonyParser>();
        } else {
            parser = std::make_unique<GenericParser>();
        }
    }
};
```

### Generation Loop (Any Backend)

```cpp
// Works for llamacpp, tensorrt, etc.
while (tokens_generated < max_tokens) {
    llama_token token = sample();

    // Check native EOS (let llama.cpp handle this)
    if (llama_vocab_is_eog(vocab, token)) {
        break;
    }

    std::string text = decode(token);

    // Parser handles format-specific logic
    if (parser->process(text)) {
        break;  // Parser said stop (<|return|> or <|call|>)
    }

    // Output content
    std::string content = parser->get_content_delta();
    if (!content.empty()) {
        output(content);
    }

    std::string reasoning = parser->get_reasoning_delta();
    if (!reasoning.empty()) {
        output_reasoning(reasoning);
    }
}
```

---

## What This Refactor Changes

1. **Think tag filtering** - Moves from `GpuBackend::output()` into GenericParser
2. **Tool call detection** - Moves from `GpuBackend::output()` into GenericParser (buffering only; parsing still in `tools/tool_parser.cpp`)
3. **Code block tracking** - Moves from `GpuBackend::output()` into GenericParser
4. **Harmony channel parsing** - Moves from `Harmony::Parser` into HarmonyParser (O(n) state machine)

## What This Refactor Does NOT Change

1. **Tool call parsing** - Still in `tools/tool_parser.cpp` (parses buffered content after generation)
2. **Prompt formatting** - Handled by templates/minja
3. **The 2048 issue** - Separate sampling/model behavior problem
4. **Backend::filter()** - Still handles CODEBLOCK vs CONTENT event labeling for frontend

---

## Files to Create/Modify

### New Files
- `backends/parser.h` - Parser abstract base class only
- `backends/generic_parser.h` - GenericParser declaration
- `backends/generic_parser.cpp` - GenericParser implementation
- `backends/harmony_parser.h` - HarmonyParser declaration
- `backends/harmony_parser.cpp` - HarmonyParser state machine implementation

### Modify
- `backends/llamacpp.cpp` - Remove 52 harmony-specific conditionals, use parser pointer
- `backends/llamacpp.h` - Add parser member
- `backends/gpu.cpp` - Simplify `output()` to just call `parser->process()` and route deltas
- `backends/gpu.h` - Remove FilterState enum and related members (moved to GenericParser)
- `backends/backend.h` - Add parser member to base class (optional)

### Remove/Deprecate
- `backends/harmony.cpp` - O(n²) parser (replace with harmony_parser.cpp)
- `backends/harmony.h` - Old parser header
- Most of `GpuBackend::output()` state machine logic (absorbed into GenericParser)

---

## Streaming Implementation

Character-by-character state machine handles both streaming and non-streaming uniformly:

```cpp
bool HarmonyParser::process(const std::string& token) {
    for (char c : token) {
        process_char(c);
        if (should_stop) return true;
    }
    return false;
}

void HarmonyParser::process_char(char c) {
    switch (char_state) {
        case CharState::NORMAL:
            if (c == '<') {
                char_state = CharState::SAW_LT;
                marker_buffer = "<";
            } else {
                emit_content(c);  // Based on current channel
            }
            break;

        case CharState::SAW_LT:
            if (c == '|') {
                char_state = CharState::SAW_LT_PIPE;
                marker_buffer += c;
            } else {
                // Not a marker, emit buffered + current
                emit_content(marker_buffer);
                emit_content(c);
                marker_buffer.clear();
                char_state = CharState::NORMAL;
            }
            break;

        case CharState::SAW_LT_PIPE:
            marker_buffer += c;
            if (c == '>') {
                // Complete marker - handle state transition
                handle_marker(marker_buffer);
                marker_buffer.clear();
                char_state = CharState::NORMAL;
            } else if (!could_be_marker_char(c)) {
                // Invalid marker, emit as content
                emit_content(marker_buffer);
                marker_buffer.clear();
                char_state = CharState::NORMAL;
            }
            // else: continue accumulating marker
            break;
    }
}
```

**Same code path for streaming and non-streaming:**
- Streaming: `process("Hel"); process("lo"); process("<|"); process("return|>");`
- Non-streaming: `process("Hello<|return|>");`

---

## Decision: Native C++ vs Rust FFI

**Chose Native C++** because:
1. Single language, simpler build
2. Full control over implementation
3. No FFI overhead or complexity
4. Rust FFI was tried before - worked but didn't fix 2048 issue
5. Linux kernel Rust concerns

The Rust harmony library (`~/src/harmony/src/encoding.rs`) has a good `StreamableParser` implementation for reference, but we'll write clean C++ instead of porting.

---

## Summary

| Problem | Solution |
|---------|----------|
| O(n²) parsing | State machine, process tokens incrementally |
| 52 harmony conditionals | Abstract Parser class, polymorphism |
| TensorRT duplication | Same parser works for any backend |
| Code organization | Format logic in parser, backend stays clean |

**Not addressed by this refactor:** The 2048 token issue (shepherd not generating `<|return|>` when llama-server/vLLM do). That's a separate investigation.

---

## Architecture Decisions

### vLLM Comparison

vLLM has three separate abstractions:
- **ReasoningParser** - Extracts thinking vs content (15+ model-specific parsers)
- **ToolParser** - Extracts tool calls (40+ model-specific parsers)
- **HarmonyContext** - Special case for GPT-OSS, uses external `openai-harmony` library

vLLM **special-cases harmony** with its own Context type, not as a pluggable parser. Shepherd's approach mirrors this: HarmonyParser is a special case, not just another parser in a registry.

vLLM still has scattered `if self.use_harmony:` conditionals (~12+ in serving_chat.py and serving_responses.py). Shepherd's polymorphic parser approach is actually cleaner in the generation loop.

### Parsing Layers (Unified)

After this refactor, parsing layers are cleaner:

1. **Output Parser** (this refactor) - HarmonyParser/GenericParser
   - **HarmonyParser**: Extracts content vs reasoning from `<|channel|>` markers
   - **GenericParser**: Extracts content vs reasoning from `<think>` tags, buffers tool calls
   - Both: Character-by-character state machine, code block awareness
   - Called per-token during generation

2. **Backend Filter** (`Backend::filter()`) - Simplified
   - Only handles CODEBLOCK vs CONTENT event labeling for frontend
   - No longer does tag parsing (moved to GenericParser)

3. **Tool Parser** (`tools/tool_parser.cpp`) - Unchanged
   - Parses buffered tool call content after generation
   - Tries multiple formats (JSON, XML, bracket, etc.)

**GpuBackend::output() becomes thin wrapper:**
```cpp
bool GpuBackend::output(const std::string& text) {
    if (parser->process(text)) {
        return false;  // Parser says stop
    }

    std::string content = parser->get_content_delta();
    if (!content.empty()) {
        filter(content);  // Just for CODEBLOCK labeling
    }

    std::string reasoning = parser->get_reasoning_delta();
    if (!reasoning.empty() && show_thinking && callback) {
        callback(CallbackEvent::THINKING, reasoning, "", "");
    }

    return true;
}
```

### Why Separate GenericParser Files

GenericParser gets its own `.h/.cpp` files (not header-only in `parser.h`) because:
1. Future tool parsing integration may add complexity
2. Consistent pattern with HarmonyParser
3. Avoids implicit inline concerns

---

## Existing Code Locations (Reference)

Files to study before implementing:

| File | Purpose | Notes |
|------|---------|-------|
| `backends/harmony.h` | Current O(n²) parser header | Replace with harmony_parser.h |
| `backends/harmony.cpp` | Current O(n²) parser impl | Replace with harmony_parser.cpp |
| `backends/gpu.h` | GpuBackend with FilterState | Remove FilterState, simplify |
| `backends/gpu.cpp` | GpuBackend::output() | Absorb into GenericParser |
| `backends/llamacpp.h` | LlamaCppBackend header | Has `harmony_parser`, `harmony_enabled` |
| `backends/llamacpp.cpp` | LlamaCppBackend impl | 52 harmony conditionals to remove |
| `backend.h` | Backend base class | Has `filter()` for CODEBLOCK |
| `backend.cpp` | Backend::filter() impl | Keep for CODEBLOCK labeling |
| `~/src/harmony/src/encoding.rs` | Rust StreamableParser | Reference for state machine design |

---

## Implementation Order

1. **Create parser.h** - Abstract base class
2. **Create generic_parser.h/.cpp** - Port GpuBackend::output() state machine
3. **Create harmony_parser.h/.cpp** - New O(n) state machine
4. **Update gpu.cpp** - Simplify output() to use parser
5. **Update llamacpp.cpp** - Remove harmony conditionals, use parser
6. **Remove harmony.h/.cpp** - Old parser no longer needed
7. **Update CMakeLists.txt** - Add new source files

---

## Testing Strategy

1. **Unit tests** - Test parsers in isolation
   - Feed known input sequences, verify content/reasoning deltas
   - Test partial marker handling (streaming simulation)
   - Test code block awareness

2. **Integration tests** - Test with real models
   - Non-harmony model (e.g., Llama): verify think tags extracted
   - Harmony model (GPT-OSS): verify channel parsing works
   - Compare output to current behavior (should be identical)

3. **Performance test** - Verify O(n) complexity
   - Generate long response (2048 tokens)
   - Time should be linear, not quadratic

4. **Regression** - Run existing test suite
   - `make test` should pass
   - Manual testing with CLI and API server
