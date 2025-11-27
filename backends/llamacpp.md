# LlamaCpp Backend Architecture

## Overview

The LlamaCpp backend (`backends/llamacpp.cpp`) provides text generation using the llama.cpp library. It manages conversation state in memory and interfaces with llama.cpp's KV cache for token processing.

## Class Structure

### LlamaCppBackend

**Inheritance:** Inherits from `Backend` base class

**Key Member Variables:**
- `void* model` - Pointer to llama_model
- `void* model_ctx` - Pointer to llama_context (KV cache lives here)
- `void* template_node` - Pointer to minja template for chat formatting
- `void* chat_templates` - Pointer to common_chat_templates
- `Session backend_session` - Tracks what's in KV cache (server mode only)
- `Session* current_session` - Pointer to active session (for callbacks)
- `size_t context_size` - Maximum context window in tokens
- `int system_formatted_tokens` - Token count of formatted system message
- `int current_user_formatted_tokens` - Token count of last user message
- `int last_assistant_formatted_tokens` - Token count of last assistant message

**Key Methods:**
- `initialize(Session& session)` - Load model and initialize KV cache
- `add_message(Session& session, ...)` - Add single message (CLI mode)
- `generate_from_session(const Session& session, ...)` - Generate from full session (server mode)
- `render_message(const Message& msg, bool add_generation_prompt)` - Format message through template
- `format_and_decode_message(Message& msg)` - Format and add tokens to KV cache
- `count_message_tokens(...)` - Count tokens for a message
- `generate(int max_tokens, StreamCallback callback)` - Generate tokens from current KV state

## Session Objects

### Session Structure
- `std::vector<Message> messages` - Ordered array of conversation messages
- `std::string system_message` - System prompt (NOT stored in messages array)
- `std::vector<Tool> tools` - Available tools for this session
- `int total_tokens` - Cumulative token count
- `int last_user_message_index` - Index of last user message
- `int last_user_message_tokens` - Token count of last user message
- `int last_assistant_message_index` - Index of last assistant message
- `int last_assistant_message_tokens` - Token count of last assistant message

### Message Structure
- `Type type` - Enum: SYSTEM, USER, ASSISTANT, TOOL
- `std::string content` - Message content
- `int tokens` - Formatted token count (includes template overhead)
- `std::string tool_name` - For tool messages
- `std::string tool_call_id` - For tool messages

### backend_session
- Used ONLY in server mode (`generate_from_session()`)
- Represents the exact state of KV cache
- Contains messages that have been decoded to KV cache
- Used for prefix comparison with incoming sessions

### current_session
- Pointer to the active Session object
- Set during `add_message()` or `generate_from_session()`
- Used by eviction callbacks to know which session to modify

## Operating Modes

### CLI Mode

**Entry Point:** `add_message(Session& session, Message::Type type, const std::string& content, ...)`

**Flow:**
1. Count tokens: `count_message_tokens(type, content, tool_name, tool_id)`
2. Create Message object with token count
3. Format and decode: `format_and_decode_message(msg)`
4. If decode succeeds, add message to `session.messages`
5. Update `session.total_tokens`
6. If USER message, call `generate()` to produce response
7. Return Response object

**State Management:**
- `current_session` points to `&session`
- No `backend_session` used
- Session passed to `add_message()` is modified directly

### Server Mode

**Entry Point:** `generate_from_session(const Session& session, int max_tokens, StreamCallback callback)`

**Flow:**
1. Compare `session.messages` with `backend_session.messages`
2. Find matching prefix (messages with same role and content)
3. Clear diverged messages from KV cache if needed
4. Set `current_session = &backend_session`
5. Add system message if not cached
6. Loop through NEW messages (from matching_prefix onward):
   - Count tokens if not set
   - Call `format_and_decode_message(msg_copy)`
   - Add to `backend_session.messages`
7. Update `backend_session.tools` and `backend_session.system_message`
8. Call `generate()` to produce response
9. Return Response object

**Prefix Matching:**
- Compares `session.messages[i]` with `backend_session.messages[i + offset]`
- offset accounts for system message in backend_session
- Stops at first mismatch of role or content
- Uses matched count to determine which messages are already cached

**Divergence Handling:**
- If `backend_session` has more messages than matched
- Calculate token position: sum of `message.tokens` for kept messages
- Clear KV cache from that position: `llama_memory_seq_rm(mem, 0, clear_from_pos, -1)`
- Remove messages from `backend_session.messages`

## Message Formatting

### render_message()

**Location:** Line 964

**Purpose:** Format a message through the chat template abstraction

**Implementation:**
- Uses `chat_template->format_message(msg)` to format the message
- ChatTemplate handles model-specific formatting (ChatML, Llama3, GLM4, etc.)
- Adds generation prompt if requested: `chat_template->get_generation_prompt()`
- No more hard-coded format strings - all handled by ChatTemplate classes

**Returns:** Formatted string ready for tokenization

**See:** `backends/chat_template.md` for details on ChatTemplate system

### format_and_decode_message()

**Location:** Line 1038

**Purpose:** Format message and add tokens to KV cache

**Flow:**
1. Check if assistant message (line 1037)
   - If YES: use raw content (no template)
   - If NO: call `render_message(msg, false)`
2. Get llama_context and llama_vocab
3. Tokenize rendered string: `llama_tokenize()`
4. Update `msg.tokens` with actual token count
5. Check if message fits in context
6. Check if eviction needed
7. Decode tokens to KV cache: `llama_batch_decode()`
8. Return success/failure

**Assistant Message Handling:**
- Never run through template
- Use `msg.content` directly
- Prevents "think tag stripping"

### count_message_tokens()

**Location:** Line 100

**Purpose:** Count tokens for a message before decoding

**Implementation:**
- Checks if model/chat_templates initialized
- Converts Message::Type to role string
- Builds `common_chat_msg` for the message
- Uses `common_chat_format_single()` with conversation context
- Tokenizes through llama_tokenize
- Returns token count

**Used For:**
- Pre-calculating token requirements
- Context limit checks
- API response metrics

## Chat Template System

### Initialization

**In initialize()** (line 1612-1677):
1. Get chat template string from model: `llama_model_meta_val_str()`
2. Save to `/tmp/shepherd_chat_template.jinja` for debugging
3. Initialize chat_templates: `common_chat_templates_init(model, ...)`
4. Parse with minja: `minja::Parser::parse()`
5. Store in `template_node` pointer (kept for MinjaTemplate fallback)
6. Detect model family from template patterns: `Models::detect_from_chat_template()`
7. **Create ChatTemplate instance**: `ChatTemplateFactory::create(template_text, model_config, template_node)`
8. Store in `chat_template` member

### ChatTemplate Abstraction

**Purpose:** Abstract chat template formatting into model-specific classes

**Factory Pattern:**
- `ChatTemplateFactory::create()` detects model family and instantiates appropriate template class
- Supports: ChatMLTemplate (Qwen), Llama3Template, GLM4Template, MinjaTemplate (fallback)

**Usage:**
- `chat_template->format_message(msg)` - Format user/assistant/tool messages
- `chat_template->format_system_message(content, tools)` - Format system message with tools
- `chat_template->get_generation_prompt()` - Get assistant start tag
- `chat_template->get_assistant_end_tag()` - Get assistant end tag

**Benefits:**
- No hard-coded format strings in backend logic
- Easy to add new template types
- Fallback to minja for unknown templates
- Consistent formatting across all message types

**See:** `backends/chat_template.md` for complete ChatTemplate documentation

### Model Family Detection

**Detected from template** (line 1667):
- Uses `Models::detect_from_chat_template(chat_template_text, model_path)`
- Checks for specific patterns in template string
- Sets `model_config.family` (LLAMA_3_X, QWEN_2_X, QWEN_3_X, GLM_4, etc.)
- Sets `model_config.tool_result_role` ("tool", "ipython", or "observation")
- Sets flags: `uses_eom_token`, `uses_python_tag`, `uses_builtin_tools_array`
- Returns ModelConfig used by ChatTemplateFactory

## KV Cache Management

### Token Decoding

**Process:**
1. Create llama_batch from tokens
2. Set sequence id (always 0)
3. Call `llama_batch_decode(ctx, batch)`
4. Tokens now in KV cache at current position

### Position Tracking

- Each message has `tokens` count
- KV position = sum of all previous message tokens
- Used for eviction and clearing operations

### Eviction

**Trigger:** Context full callback

**Process:**
1. `evict_to_free_space()` called
2. Determine what to keep (system, last user, last assistant)
3. Calculate token position to clear from
4. Clear KV cache: `llama_memory_seq_rm()`
5. Remove messages from session

### Memory Operations

**llama_memory_t:** Unified memory manager
- `llama_get_memory(ctx)` - Get memory handle
- `llama_memory_seq_rm(mem, seq_id, pos_start, pos_end)` - Remove tokens
- Position -1 means "to end"

## Generation

### generate()

**Location:** Line 474

**Signature:** `std::string generate(int max_tokens = 0, StreamCallback callback = nullptr)`

**Purpose:** Generate tokens from current KV state (internal method)

**Parameters:**
- `max_tokens` - Maximum tokens to generate (0 = use default)
- `callback` - Optional streaming callback for token-by-token output

**Flow:**
1. Extract tool call markers from chat template
2. Log KV cache state (messages in cache)
3. Add generation prompt to KV cache (e.g., `<|im_start|>assistant\n`)
4. Call `run_inference()` with callback
5. Add closing tag to KV cache (e.g., `<|im_end|>\n`)
6. Return generated text

**Used by:**
- CLI mode: `add_message()` calls `generate()` without callback
- Server mode: `generate_from_session()` calls `generate()` with callback for streaming

### run_inference()

**Location:** Line 714

**Signature:** `std::string run_inference(const std::string& prompt_text, int max_tokens, bool suppress_streaming = false, StreamCallback callback = nullptr)`

**Purpose:** Core token generation loop using llama.cpp

**Parameters:**
- `prompt_text` - Text to generate from (usually empty since KV cache is pre-filled)
- `max_tokens` - Maximum tokens to generate
- `suppress_streaming` - If true, don't write to terminal (set in server mode)
- `callback` - Optional streaming callback for API server streaming

**Flow:**
1. Calculate available tokens (context - used)
2. Cap max_tokens if needed
3. Create sampling context: `common_sampler_init()`
4. Generation loop until stop condition:
   - Sample next token: `common_sampler_sample()`
   - Check for EOS/EOG tokens (break if found)
   - Convert token to text: `llama_token_to_piece()`
   - Accumulate text in response string
   - **Call streaming callback if provided** (line 882-891)
   - Write to terminal if not suppressed (line 894-896)
   - Decode token into KV cache
   - Check for cancellation (Escape key)
5. Return accumulated response text

**Sampling:**
- Uses `common_sampler` from llama.cpp
- Configured with temperature, top_p, top_k, etc.
- Supports penalties (repeat, frequency, presence)

**Streaming:**
- Two streaming mechanisms:
  1. **Terminal streaming** - Writes to `tio` (TerminalIO) unless `suppress_streaming=true`
  2. **Callback streaming** - Calls provided callback with delta for each token
- Callback signature: `bool callback(const std::string& delta, const std::string& accumulated, const Response& partial)`
- Callback returns bool (false = stop generation)
- Used by API server for SSE streaming to clients

## Initialization Flow

### initialize(Session& session)

**Purpose:** Load model and prepare for inference

**Steps:**
1. Load model: `llama_model_load(model_path, params)`
2. Set context size from model metadata
3. Configure GPU layers (auto-detect or specified)
4. Configure multi-GPU if multiple devices
5. Create context: `llama_backend_init_ctx(model, ctx_params)`
6. Initialize chat templates
7. Detect model family
8. Add system message to session if provided

**GPU Configuration:**
- Detects available CUDA devices
- Supports pipeline parallelism (PP) and tensor parallelism (TP)
- Split modes: LAYER (PP) or ROW (TP)
- Distributes model across devices

**Batch Size:**
- Default: 512 tokens
- GPU: increased to 2048
- Used for prompt processing

## Tool Handling

### System Message with Tools

**Format via Models::format_system_message():**
1. Get base system message
2. If tools present, append tool descriptions
3. Format as JSON schema or plain text
4. Return formatted string

**Tool Invocation:**
- Model generates tool_call in response
- Parser extracts tool name and arguments
- Shepherd executes tool
- Result added as TOOL message
- Conversation continues

## Debug Output

### Response Debug File

**Location:** `/tmp/shepherd_response_debug.txt`

Raw model responses are always written to this file for debugging. Each response is logged with timestamp and length:

```
=== Response at <timestamp> ===
<raw response content including think tags if present>
=== End (length: <bytes>) ===
```

This is useful for debugging thinking model behavior and verifying what the model actually outputs before any filtering.

## File Locations

- `backends/llamacpp.cpp` - Implementation
- `backends/llamacpp.h` - Header
- `llama.cpp/` - Submodule dependency
- `/tmp/shepherd_chat_template.jinja` - Extracted template (debug)
- `/tmp/shepherd_response_debug.txt` - Raw response log (debug)
