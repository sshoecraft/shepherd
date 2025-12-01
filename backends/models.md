# Models Architecture

## Overview

The models module (`models.h`, `models.cpp`) provides centralized model family detection and configuration. It determines how to format prompts, handle tool calls, and parse responses based on the model being used.

## Model Families

Defined in `ModelFamily` enum:

| Family | Models | Chat Format | Tool Role |
|--------|--------|-------------|-----------|
| `LLAMA_2_X` | Llama 1.x, 2.x, TinyLlama | `[INST]...[/INST]` | tool |
| `LLAMA_3_X` | Llama 3.0-3.3 | `<\|start_header_id\|>...<\|end_header_id\|>` | ipython |
| `QWEN_2_X` | Qwen 2.x | `<\|im_start\|>...<\|im_end\|>` | tool |
| `QWEN_3_X` | Qwen 3.x, MindLink | `<\|im_start\|>...<\|im_end\|>` | tool |
| `GLM_4` | GLM-4.x | `<\|assistant\|>` | observation |
| `MISTRAL` | Mistral | `[INST]...[/INST]` | tool |
| `GENERIC` | Unknown models | Basic format | tool |

## Detection Methods

### Primary: Chat Template Analysis

`Models::detect_from_chat_template()` analyzes the Jinja template for model-specific patterns:

- `<|im_start|>` + `<|im_end|>` → Qwen family
- `<|start_header_id|>` → Llama 3.x
- `<|observation|>` → GLM-4
- `[INST]` → Llama 2.x or Mistral

### Secondary: config.json Analysis

`Models::detect_from_config_file()` reads the model's config.json and checks:

1. `architecture` field (e.g., "LlamaForCausalLM", "Qwen2ForCausalLM")
2. `model_type` field (e.g., "llama", "qwen2", "chatglm")
3. `qwen_type` field (TensorRT-LLM Qwen format)

For Llama models, it checks for Llama 3.x tokens (`<|begin_of_text|>`, `<|eom_id|>`) to distinguish from Llama 2.x.

### Tertiary: Path Analysis

`Models::detect_from_model_path()` checks filename patterns:

- `llama-3`, `llama3` → Llama 3.x
- `qwen2`, `qwen-2` → Qwen 2.x
- `mindlink`, `qwen3` → Qwen 3.x
- `glm-4`, `glm4` → GLM-4

## ModelConfig Structure

Each model family has a factory method returning a configured `ModelConfig`:

```cpp
ModelConfig::create_llama_2x()   // Llama 1.x, 2.x, TinyLlama
ModelConfig::create_llama_3x()   // Llama 3.x with tool support
ModelConfig::create_qwen_2x()    // Qwen 2.x
ModelConfig::create_qwen_3x()    // Qwen 3.x with optional thinking mode
ModelConfig::create_glm_4()      // GLM-4.x
ModelConfig::create_generic()    // Fallback
```

Key configuration fields:

- `family` - ModelFamily enum value
- `tool_result_role` - Role name for tool results ("ipython", "tool", "observation")
- `uses_eom_token` - Whether model uses `<|eom_id|>` for continued tool calls
- `uses_python_tag` - Whether to use `<|python_tag|>` format
- `supports_thinking_mode` - Model can output `<think>` blocks
- `assistant_start_tag` / `assistant_end_tag` - Message boundary tokens

## API Model Detection

For API backends (OpenAI, Anthropic, etc.), `Models::detect_from_api_model()` looks up model metadata including:

- Context window size
- Maximum output tokens
- Supported capabilities (vision, audio, function calling)
- Provider-specific headers

## Key Files

- `models.h` - ModelFamily enum, ModelConfig struct, factory methods
- `models.cpp` - Detection logic implementation

## Adding New Model Families

1. Add enum value to `ModelFamily` in `models.h`
2. Add `create_<family>()` factory method in `ModelConfig` struct
3. Add detection patterns in `models.cpp`:
   - `detect_from_template_content()` for chat template patterns
   - `detect_from_config_file()` for architecture/model_type mappings
   - `detect_from_path_analysis()` for filename patterns

All detection logic is centralized in `models.cpp` - backends just call the detection functions.
