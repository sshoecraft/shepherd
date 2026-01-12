# StreamParser Module

## Overview

The StreamParser module provides a unified interface for processing generated tokens during streaming output. It extracts content, reasoning/thinking, and tool calls from model responses.

## Architecture

```
                    StreamParser::Parser (abstract base)
                           /          \
                          /            \
              GenericParser          HarmonyParser
            (non-harmony models)   (GPT-OSS models)
```

## Components

### parser.h
Abstract base class `StreamParser::Parser` with interface:
- `process(token)` - Process a token, returns true if generation should stop
- `get_content_delta()` - Get user-facing content since last call
- `get_reasoning_delta()` - Get thinking/reasoning since last call
- `get_tool_calls()` - Get completed tool calls since last call
- `reset()` - Reset parser state for new generation
- `flush()` - Flush any buffered content at end of generation

Factory function `create_parser()` creates the appropriate parser based on model capabilities.

### generic_parser.h/.cpp
For non-harmony models (most LLMs). Handles:
- Tool call detection (`<tool_call>`, JSON `{...}`, etc.)
- Thinking tag extraction (`<think>`, `</think>`, etc.)
- Code block awareness (don't parse tags inside ```)

State machine ported from `GpuBackend::output()`.

### harmony_parser.h/.cpp
O(n) character-by-character state machine for GPT-OSS harmony format. Handles:
- Channel routing (`<|channel|>analysis` -> reasoning, `<|channel|>final` -> content)
- Stop token detection (`<|return|>`, `<|call|>`)
- Message boundaries (`<|start|>`, `<|message|>`, `<|end|>`)

Replaces the O(n^2) parser in `harmony.cpp` which re-parsed the entire buffer on every token.

## Integration

The parser is managed by `GpuBackend`:
- Created in `reset_output_state()` based on model capabilities
- `HarmonyParser` for models with `has_channels` capability
- `GenericParser` for all other models

`LlamaCppBackend::run_inference()` uses the parser:
```cpp
bool should_stop = parser->process(token);
std::string content = parser->get_content_delta();
std::string reasoning = parser->get_reasoning_delta();
auto tool_calls = parser->get_tool_calls();
```

## Files

| File | Purpose |
|------|---------|
| backends/parser.h | Abstract base class and factory |
| backends/generic_parser.h | GenericParser declaration |
| backends/generic_parser.cpp | GenericParser implementation |
| backends/harmony_parser.h | HarmonyParser declaration |
| backends/harmony_parser.cpp | HarmonyParser O(n) state machine |
| backends/harmony.h | Legacy O(n^2) parser (kept for tests) |
| backends/harmony.cpp | Legacy parser implementation |

## History

- 2026-01-10: Created parser abstraction to replace scattered harmony conditionals
  - Added abstract Parser base class
  - Ported GpuBackend::output() state machine to GenericParser
  - Created O(n) HarmonyParser replacing O(n^2) Harmony::Parser
  - Updated LlamaCppBackend to use unified parser interface
