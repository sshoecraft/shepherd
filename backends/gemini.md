# Gemini Backend

## Overview

The Gemini backend (`gemini.cpp`, `gemini.h`) provides integration with Google's Gemini AI models via the Gemini API.

## Supported Models

- Gemini 2.0 Flash (gemini-2.0-flash-001)
- Gemini 1.5 Pro (gemini-1.5-pro)
- Gemini 1.5 Flash (gemini-1.5-flash)
- Gemini Pro (gemini-pro)

## Features

### Streaming
- Uses SSE format via `streamGenerateContent?alt=sse` endpoint
- Parses `candidates[].content.parts[].text` for content chunks

### Tool/Function Calling
- Full function calling support via `functionDeclarations`
- Parses `functionCall` responses with name and args
- Emits `TOOL_CALL` callback events for each function call

### Token Counting
- Uses adaptive EMA-based estimation (inherited from ApiBackend)
- API returns token counts in `usageMetadata`

## API Format

### Request Structure
```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "..."}]},
    {"role": "model", "parts": [{"text": "..."}]}
  ],
  "systemInstruction": {
    "parts": [{"text": "..."}]
  },
  "tools": [{
    "functionDeclarations": [
      {"name": "...", "description": "...", "parameters": {...}}
    ]
  }],
  "generationConfig": {
    "temperature": 0.7,
    "topP": 0.95,
    "topK": 40,
    "maxOutputTokens": 8192
  }
}
```

### Response Structure (Streaming)
```
data: {"candidates":[{"content":{"parts":[{"text":"..."}]}}],"usageMetadata":{...}}
```

### Function Call Response
```json
{
  "candidates": [{
    "content": {
      "parts": [{
        "functionCall": {
          "name": "get_time",
          "args": {}
        }
      }]
    }
  }]
}
```

## Configuration

### Provider Settings
```json
{
  "type": "gemini",
  "model": "gemini-2.0-flash-001",
  "api_key": "YOUR_API_KEY",
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_tokens": 8192
}
```

### Context Sizes (from models database)
| Model | Context Window |
|-------|---------------|
| gemini-2.0-flash-001 | 1,048,576 |
| gemini-1.5-pro | 2,097,152 |
| gemini-1.5-flash | 1,048,576 |
| gemini-pro | 32,768 |

## Implementation Notes

### Role Mapping
- Shepherd `USER` → Gemini `user`
- Shepherd `ASSISTANT` → Gemini `model`
- Shepherd `SYSTEM` → `systemInstruction` (separate field)
- Shepherd `TOOL_RESPONSE` → Gemini `functionResponse`

### Callback Events
The backend emits these callback events:
- `CONTENT` - Text content from response
- `TOOL_CALL` - Function call detected (name, id, args JSON)
- `STOP` - Generation complete
- `ERROR` - Error occurred

## History

- v2.7.0: Fixed TOOL_CALL callback emission in generate_from_session()
  - Function calls now properly emit callback events
  - Enables nested tool execution in APIToolAdapter

- v2.6.1: Added streaming support via add_message_stream()

- v2.5.0: Initial Gemini backend implementation
