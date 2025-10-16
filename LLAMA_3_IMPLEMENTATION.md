# Llama 3.1 Prompt Format Implementation

## Summary

Implemented model-specific prompt format detection and configuration to properly support Llama 3.1's tool calling with `ipython` role and `<|eom_id|>` token handling.

## Changes Made

### 1. Model Configuration System (`model_config.h`)

Created a new model configuration system with:
- `ModelFamily` enum supporting Llama 3.x, GLM-4.x, Qwen 2.x, and others
- `ModelConfig` struct with model-specific settings:
  - `tool_result_role`: "ipython" for Llama 3.x, "tool" for others, "observation" for GLM-4.x
  - `uses_eom_token`: Enables `<|eom_id|>` detection for continued tool execution
  - `uses_python_tag`: For `<|python_tag|>` format support
  - `uses_builtin_tools_array`: For `builtin_tools` template variable

Factory methods for common model families:
```cpp
ModelConfig::create_llama_3x("3.1")
ModelConfig::create_glm_4("4.5")
ModelConfig::create_qwen_2x("2.5")
```

### 2. Model Detection (`backends/tensorrt.cpp`)

Added `TensorRTBackend::detect_model_family()` with three-tier detection:

**Primary: Chat Template Analysis**
- Most reliable method
- Llama 3.x: Detects `"Environment: ipython"` + `"<|eom_id|>"`
- GLM-4.x: Detects `"<|observation|>"` + `"<think>"`
- Qwen 2.x: Detects `"<|im_start|>"`

**Fallback: config.json Parsing**
- Reads `model_type` field
- Cross-references with special tokens

**Final: Generic Configuration**
- Safe defaults for unknown models

### 3. Message Role Mapping (`context_manager.h`)

Extended `Message` class with:
```cpp
std::string get_role_for_model(const ModelConfig& config) const {
    if (type != TOOL) {
        return get_role();  // Standard roles
    }
    return config.tool_result_role;  // Model-specific: "ipython", "tool", etc.
}
```

### 4. Context Manager Integration (`backends/tensorrt.h/cpp`)

**TensorRTContextManager:**
- Added `model_config_` field
- Added `set_model_config()` method
- Modified `get_context_for_inference()` to use `get_role_for_model()`

**TensorRTBackend::initialize():**
- Calls `detect_model_family()` after loading chat template
- Passes config to context manager
- Logs detected configuration

### 5. Tool Execution Flow (`backends/tensorrt.cpp`)

**In `TensorRTBackend::generate()`:**
- Detects `<|eom_id|>` token when `model_config_.uses_eom_token == true`
- Strips `<|eom_id|>` from response
- Logs continuation signal
- Main loop in `main.cpp` continues tool execution automatically

## How It Works

### Llama 3.1 Flow:

1. **Initialization:**
   ```
   Load tokenizer_config.json → Extract chat_template
   ↓
   detect_model_family() → Finds "Environment: ipython"
   ↓
   Returns ModelConfig with tool_result_role="ipython", uses_eom_token=true
   ↓
   Passes config to TensorRTContextManager
   ```

2. **Message Rendering:**
   ```
   Tool result message (type=TOOL) created
   ↓
   get_role_for_model(config) → Returns "ipython"
   ↓
   Template renders: <|start_header_id|>ipython<|end_header_id|>
   ```

3. **Tool Execution Loop:**
   ```
   User: "What's the weather?"
   ↓
   Model generates: {"name": "get_weather", ...}<|eom_id|>
   ↓
   Backend strips <|eom_id|>, returns JSON
   ↓
   main.cpp detects tool call → Executes tool
   ↓
   Adds result with role="ipython"
   ↓
   Model continues → Final response<|eot_id|>
   ```

## Llama 3.1 Specific Features

### Special Tokens
- `<|begin_of_text|>`: Start of prompt (BOS)
- `<|start_header_id|>role<|end_header_id|>`: Role markers
- `<|eom_id|>`: End of message (continue execution)
- `<|eot_id|>`: End of turn (final response)
- `<|python_tag|>`: Built-in tool marker (optional)

### System Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: [current]

[system instructions]<|eot_id|>
```

### Tool Result Format
```
<|start_header_id|>ipython<|end_header_id|>

[tool result]<|eot_id|>
```

## Testing

To test with Llama 3.1:

1. Load a Llama 3.1 model (8B, 70B, or 405B)
2. Check logs for: "Detected Llama 3.x model family from chat template"
3. Verify: "tool_result_role=ipython, uses_eom_token=true"
4. Test tool calling:
   ```
   User: "What's 2+2 using Python?"
   Expected: Model calls tool → Executes → Returns answer
   ```
5. Check rendered prompts in debug logs for `<|start_header_id|>ipython<|end_header_id|>`

## Benefits

1. **Automatic Detection**: No manual configuration needed
2. **Extensible**: Easy to add new model families
3. **Robust**: Multiple fallback detection methods
4. **Template-Driven**: Uses existing chat template as source of truth
5. **Backward Compatible**: Generic fallback for unknown models

## Future Extensions

### Adding New Models

To add support for a new model family:

1. Add enum to `ModelFamily`
2. Add detection pattern in `detect_model_family()`
3. Create factory method in `ModelConfig`
4. Done!

Example for Mistral:
```cpp
// In detect_model_family():
if (chat_template_text_.find("[INST]") != std::string::npos) {
    return ModelConfig::create_mistral();
}

// In model_config.h:
static ModelConfig create_mistral() {
    return ModelConfig{
        .family = ModelFamily::MISTRAL,
        .tool_result_role = "tool",
        .uses_eom_token = false,
        // ...
    };
}
```

## References

- Llama 3.1 Docs: /home/steve/llama-3.1.pdf
- Chat Template: llama.cpp/models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja
- Model Config: model_config.h
- Implementation: backends/tensorrt.cpp:295-389
