# OpenAI Backend

## Overview

The OpenAI backend provides an OpenAI-compatible API client for connecting to various LLM providers including OpenAI, Anthropic (via proxy), local servers (vLLM, TRT-LLM, llama.cpp server), and other compatible APIs.

## Error Handling

The backend parses error responses from multiple formats:

### OpenAI Format
```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```
or
```json
{"error": "error string"}
```

### TRT-LLM/vLLM Format
```json
{"object": "error", "message": "...", "type": "..."}
```

The message is at the root level, not nested under "error".

## Token Eviction

For context overflow errors, the backend parses various error message formats to extract token counts:
- OpenAI classic: "resulted in X tokens"
- vLLM: "max_tokens is too large... your request has X input tokens"
- OpenAI detailed: "requested X tokens"

## History

- v2.5.1: Added support for TRT-LLM/vLLM error message format (root-level "message")
