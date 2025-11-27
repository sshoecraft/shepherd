# ChatTemplate System Architecture

## Overview

The ChatTemplate system (`backends/chat_template.h` and `backends/chat_template.cpp`) provides an abstraction layer for formatting chat messages according to different model-specific templates. It replaces hard-coded format strings with a clean class hierarchy that's easy to extend.

## Purpose

Different LLM models use different chat template formats (ChatML, Llama3, GLM4, etc.). Rather than embedding format strings throughout the backend code, the ChatTemplate system:
- Encapsulates model-specific formatting logic in dedicated classes
- Provides a uniform interface for all template operations
- Falls back to minja template rendering for unknown formats
- Makes it easy to add support for new template types

## Class Hierarchy

### Base Class: ChatTemplate

**Location:** `backends/chat_template.h:12`

**Purpose:** Pure virtual interface defining the contract for all chat templates

**Methods:**
```cpp
virtual std::string format_message(const Message& msg) const = 0;
virtual std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const = 0;
virtual std::string get_generation_prompt() const = 0;
virtual std::string get_assistant_end_tag() const = 0;
virtual ModelFamily get_family() const = 0;
```

### Concrete Template Classes

#### ChatMLTemplate

**Location:** `backends/chat_template.cpp:13`

**Used for:** Qwen 2.x, Qwen 3.x, MindLink models

**Format:**
- Message: `<|im_start|>{role}\n{content}<|im_end|>\n`
- System with tools: XML-based tool descriptions in `<tools></tools>` tags
- Generation prompt: `<|im_start|>assistant\n`
- End tag: `<|im_end|>\n`

**Tool Format:**
- Tools listed as JSON objects within `<tools>` XML tags
- Tool calls expected in `<tool_call>` XML tags with JSON content
- System prompt explains tool usage format

**Returns:** `ModelFamily::QWEN_2_X`

#### Qwen3ThinkingTemplate

**Location:** `backends/chat_template.cpp:65`

**Used for:** Qwen 3.x thinking models (including MindLink)

**Inherits from:** ChatMLTemplate

**Purpose:** Handles thinking/reasoning models by injecting an empty `<think>` block in the generation prompt when thinking is disabled, preventing the model from entering thinking mode.

**Overridden Methods:**
- `get_generation_prompt()`: Returns `<|im_start|>assistant\n<think>\n\n</think>\n\n` when `config->thinking` is false (disabled), or standard `<|im_start|>assistant\n` when thinking is enabled
- `get_family()`: Returns `ModelFamily::QWEN_3_X`

**Behavior:**
- When thinking is disabled (default): Pre-injects empty think block to tell the model "thinking is done, just respond"
- When thinking is enabled (`--thinking` flag or `thinking=true` in config): Lets the model do its natural thinking

**Returns:** `ModelFamily::QWEN_3_X`

#### Llama3Template

**Location:** `backends/chat_template.cpp:64`

**Used for:** Llama 3.x models

**Format:**
- Message: `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`
- Tool role: Uses "ipython" for tool results (not "tool")
- System with tools: BFCL-style JSON schema descriptions
- Generation prompt: `<|start_header_id|>assistant<|end_header_id|>\n\n`
- End tag: `<|eot_id|>`

**Tool Format:**
- Tools listed as function name + description + JSON schema
- Instructs model to respond with JSON object: `{"name": "...", "parameters": {...}}`
- Emphasizes only calling functions when actually needed

**Returns:** `ModelFamily::LLAMA_3_X`

#### GLM4Template

**Location:** `backends/chat_template.cpp:123`

**Used for:** GLM-4.x models

**Format:**
- Message: `<|{role}|>\n{content}`
- Tool role: Uses "observation" for tool results (not "tool")
- System: Chinese default message with English instruction
- Generation prompt: `<|assistant|>\n`
- End tag: Empty string (no end tag)

**Tool Format:**
- Tools described in Chinese (可用工具)
- Flat JSON format: `{"name": "tool_name", "parameters": {...}}`
- Includes file reading optimization instructions in Chinese

**Returns:** `ModelFamily::GLM_4`

#### MinjaTemplate

**Location:** `backends/chat_template.cpp:169`

**Used for:** Unknown/unsupported models (fallback)

**Constructor:**
```cpp
MinjaTemplate(const std::string& template_text, void* template_node_ptr)
```

**Implementation:**
- Stores reference to minja template_node from llama.cpp
- For assistant messages: Returns raw content (no template processing)
- For other messages: Renders through minja template with full context
- Falls back to generic format if template_node is null

**Format:**
- Uses minja template rendering with context (strftime_now, date_string, messages array)
- Generic fallback: `{role}: {content}\n\n`

**Tool Handling:**
- Converts Session::Tool to minja::Value format
- Round-trips nlohmann::json through string to avoid type ambiguity
- Passes tools array to template context

**Returns:** `ModelFamily::GENERIC`

### ChatTemplateFactory

**Location:** `backends/chat_template.cpp:309`

**Purpose:** Factory class to create appropriate ChatTemplate instance based on model family

**Method:**
```cpp
static std::unique_ptr<ChatTemplate> create(
    const std::string& template_text,
    const ModelConfig& config,
    void* template_node_ptr = nullptr
);
```

**Logic:**
- Switches on `config.family` to determine template type
- Creates ChatMLTemplate for QWEN_2_X
- Creates Qwen3ThinkingTemplate for QWEN_3_X when `config.supports_thinking_mode` is true
- Creates ChatMLTemplate for QWEN_3_X when thinking mode is not supported
- Creates Llama3Template for LLAMA_3_X
- Creates GLM4Template for GLM_4
- Falls back to MinjaTemplate for all other families

**Returns:** `std::unique_ptr<ChatTemplate>` to appropriate subclass

## Integration with LlamaCppBackend

### Initialization

**Location:** `backends/llamacpp.cpp:1676`

```cpp
chat_template = ChatTemplates::ChatTemplateFactory::create(chat_template_text, model_config, template_node);
```

**Flow:**
1. Model loaded, template extracted from model metadata
2. ModelConfig detected from template patterns
3. Factory creates appropriate ChatTemplate instance
4. Stored in `chat_template` member variable
5. Used for all subsequent message formatting

### Usage in render_message()

**Location:** `backends/llamacpp.cpp:972`

```cpp
std::string formatted = chat_template->format_message(msg);
if (add_generation_prompt && msg.get_role() == "user") {
    formatted += chat_template->get_generation_prompt();
}
```

**Replaces:** Previous hard-coded ChatML format and minja template calls

### Usage in generate()

**Location:** `backends/llamacpp.cpp:567, 617`

```cpp
// Get generation prompt
std::string generation_prompt = chat_template->get_generation_prompt();

// Get assistant end tag
std::string assistant_end_tag = chat_template->get_assistant_end_tag();
```

**Replaces:** `model_config.assistant_start_tag` and `model_config.assistant_end_tag`

### Usage in system message formatting

**Location:** `backends/llamacpp.cpp:1258, 1711, 1756`

```cpp
// Convert tools and format system message
std::vector<Session::Tool> tools = session.tools.empty() ?
    convert_registry_to_session_tools(ToolRegistry::instance()) : session.tools;
std::string formatted_system = chat_template->format_system_message(session.system_message, tools);
```

**Replaces:** `Models::format_system_message()` calls

## Helper Functions

### convert_registry_to_session_tools()

**Location:** `backends/llamacpp.cpp:24`

**Purpose:** Convert ToolRegistry tools to Session::Tool format for ChatTemplate

**Flow:**
1. Iterate through all tools in ToolRegistry
2. Extract name, description, and parameters schema
3. Build Session::Tool object with JSON schema
4. Return vector of Session::Tool

**Used by:** System message formatting in both CLI and server modes

## Data Flow

### Message Formatting Flow

1. **Message Creation:** Backend receives message (user/assistant/tool)
2. **Template Selection:** ChatTemplate already instantiated based on model family
3. **Formatting:** Call `chat_template->format_message(msg)`
4. **Template Logic:** Concrete template class applies model-specific format
5. **Tokenization:** Formatted string passed to llama.cpp tokenizer
6. **KV Cache:** Tokens decoded into KV cache

### Tool Integration Flow

1. **Tool Registry:** Tools registered in ToolRegistry singleton
2. **Conversion:** `convert_registry_to_session_tools()` creates Session::Tool vector
3. **System Formatting:** `chat_template->format_system_message(content, tools)`
4. **Model-Specific:** Each template formats tools according to its requirements
5. **Prompt Creation:** System message with embedded tool descriptions

## Adding New Templates

To add support for a new chat template format:

1. **Create Template Class:**
   - Add new class in `backends/chat_template.cpp`
   - Inherit from `ChatTemplate`
   - Implement all virtual methods

2. **Update Factory:**
   - Add new ModelFamily enum to `backends/models.h`
   - Add case in `ChatTemplateFactory::create()` switch statement

3. **Update Detection:**
   - Add pattern detection in `Models::detect_from_chat_template()`
   - Add ModelConfig factory method in `backends/models.h`

4. **Test:**
   - Verify message formatting
   - Test system message with tools
   - Verify generation prompt and end tags

## File Locations

- `backends/chat_template.h` - Header with class declarations
- `backends/chat_template.cpp` - Implementation of all template classes
- `backends/llamacpp.h` - LlamaCppBackend header (chat_template member)
- `backends/llamacpp.cpp` - Integration with backend (usage points)
- `backends/models.h` - ModelFamily enum, ModelConfig struct
- `backends/models.cpp` - Model detection and configuration

## Important Notes

1. **Assistant Messages:** Always use raw content, never run through template (prevents think tag stripping)
2. **System Messages:** Always run through template (includes tool descriptions)
3. **Tool Format:** Each model family has its own tool description format
4. **JSON Conversion:** MinjaTemplate requires round-trip through string to avoid nlohmann::json vs minja::json ambiguity
5. **Fallback:** MinjaTemplate provides safe fallback for unknown model types
6. **No Hard-Coding:** All format strings are in template classes, not in backend logic

## Performance Considerations

- Template selection happens once during initialization (not per-message)
- Formatting is lightweight string concatenation (except MinjaTemplate)
- MinjaTemplate rendering has overhead but only used for unknown models
- No runtime detection or dynamic dispatch beyond virtual method calls

## Version History

- **2.4.0** - Thinking model support
  - Added Qwen3ThinkingTemplate for thinking/reasoning models
  - Handles MindLink and other Qwen3 thinking model variants
  - Injects empty think block when thinking is disabled (matches llama-server behavior)
  - Factory now checks `config.supports_thinking_mode` for QWEN_3_X models

- **2.3.0** - Initial ChatTemplate system implementation
  - Added ChatMLTemplate, Llama3Template, GLM4Template, MinjaTemplate
  - Removed hard-coded ChatML format strings
  - Added ChatTemplateFactory for automatic selection
  - Integrated with LlamaCppBackend
