# Llama 3.1 PDF-Compliant Implementation

## Summary
Updated Shepherd to match the **exact** Llama 3.1 JSON tool calling format specified in Meta's official PDF (pages 10-11).

## What Changed

### Before (Incorrect)
- System message contained ALL JSON schemas
- System message was very long with detailed instructions
- First user message was just the question
- Format: `Environment: ipython` + schemas in system

### After (PDF-Compliant)
- **System message is minimal** (per PDF page 10):
  ```
  Cutting Knowledge Date: December 2023
  Today Date: 09 October 2025

  When you receive a tool call response, use the output to format an answer to the original user question.

  You are a helpful assistant with tool calling capabilities.
  ```

- **First user message contains JSON schemas** (per PDF page 10-11):
  ```
  Given the following functions, please respond with a JSON for a function call...

  {JSON schemas here}

  Question: {user's actual question}
  ```

- **Subsequent user messages** are normal (no schemas)

## Implementation Details

### 1. Minimal System Prompt (main.cpp:1082-1104)
```cpp
if (model_config.family == ModelFamily::LLAMA_3_X) {
    system_message = "Cutting Knowledge Date: December 2023\n";
    system_message += "Today Date: " + current_date + "\n\n";
    system_message += "When you receive a tool call response, use the output to format an answer to the original user question.\n\n";
    system_message += "You are a helpful assistant with tool calling capabilities.\n";

    // Generate JSON schemas for first user message
    g_llama3_tool_schemas = generate_llama3_tool_schemas(tool_descriptions, registry);
}
```

### 2. New Function: generate_llama3_tool_schemas() (main.cpp:600-651)
Generates the JSON schemas section that gets prepended to the first user message:
- Instruction text from PDF
- All tool schemas in OpenAI-compatible JSON format
- Memory tools listed first

### 3. First User Message Prepending (main.cpp:1230-1236)
```cpp
std::string final_user_message = user_input;
if (!g_llama3_tool_schemas.empty() && !g_first_user_message_sent) {
    final_user_message = g_llama3_tool_schemas + "Question: " + user_input;
    g_first_user_message_sent = true;
}
backend->add_user_message(final_user_message);
```

### 4. Updated format_tools_for_model() (main.cpp:673-681)
For Llama 3.x, only returns `Environment: ipython` (no schemas):
```cpp
if (config.family == ModelFamily::LLAMA_3_X) {
    result += "Environment: ipython\n\n";
    return result;  // Schemas go in first user message, not system
}
```

## Files Modified

1. **main.cpp**:
   - Added `g_llama3_tool_schemas` and `g_first_user_message_sent` globals (line 53-54)
   - Added `generate_llama3_tool_schemas()` function (line 600-651)
   - Updated `format_tools_for_model()` for Llama 3.x (line 673-681)
   - Minimal system prompt for Llama 3.x (line 1082-1104)
   - Prepend schemas to first user message (line 1230-1236)

## Expected Prompt Format

### System Message (sent once)
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 09 October 2025

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities.<|eot_id|>
```

### First User Message
```
<|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{
  "type": "function",
  "function": {
    "name": "search_memory",
    "description": "Search historical conversation context...",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query to find relevant information"
        },
        ...
      },
      "required": ["query"]
    }
  }
}

{...more tool schemas...}

Question: What did we discuss yesterday?<|eot_id|>
```

### Subsequent Messages
```
<|start_header_id|>assistant<|end_header_id|>
{"name": "search_memory", "parameters": {"query": "yesterday discussion"}}<|eot_id|>

<|start_header_id|>ipython<|end_header_id|>
{tool result}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Based on the search results, yesterday we discussed...<|eot_id|>
```

## Why This Matters

✅ **Matches PDF specification exactly** (pages 10-11)
✅ **Cleaner system prompt** (not polluted with long tool definitions)
✅ **Proper context management** (schemas only in first message where needed)
✅ **Better model performance** (follows training format precisely)
✅ **OpenAI-compatible** JSON schemas

## Testing

Build successful: ✅

To verify with actual model:
```bash
./build/shepherd --backend llamacpp --model /path/to/llama-3.1-8b.gguf --debug
```

Expected logs:
```
[INFO] Detected Llama 3.x model family from chat template
[DEBUG] Generated JSON schemas for Llama 3.x (XXXX chars)
[INFO] Prepended JSON tool schemas to first user message
```

## Backward Compatibility

- Non-Llama 3.x models: unchanged (use detailed system prompt with tools)
- Llama 3.x models: new PDF-compliant format
- All tool functionality: preserved

## References

- **Llama 3.1 PDF**: ~/llama-3.1.pdf (pages 10-11 for JSON tool calling)
- **Meta Docs**: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
- **Previous Implementation**: LLAMA_3_JSON_TOOLS_FINAL.md (now superseded)
