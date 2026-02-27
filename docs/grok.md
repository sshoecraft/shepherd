# Grok Backend (backends/grok.h, backends/grok.cpp)

## Overview
Dedicated backend for xAI's Grok API. Inherits from OpenAIBackend since Grok uses the OpenAI-compatible wire protocol (SSE streaming, /v1/models, OpenAI message format, tool calling).

## Architecture
- **Class**: `GrokBackend : public OpenAIBackend`
- Inherits all shared protocol code from OpenAI (streaming, response parsing, error parsing, model queries, auth)
- Overrides only the request builders with clean Grok-specific implementations

## What's Different from OpenAI
- **Reasoning format**: Grok uses `"reasoning": {"effort": "low"|"high"}` (not OpenAI's `"reasoning_effort"`)
- **No `presence_penalty`** — Grok rejects it on some models
- **No `top_k` / `repetition_penalty`** — non-standard params not sent
- **No `openai_strict` branching** — request format is fixed for Grok
- **No Azure support** — not applicable

## Reasoning
Uses Grok's native format: `"reasoning": {"effort": "low"|"high"}`
- Supported by: grok-3-mini, grok-4-fast-reasoning, grok-4-1-fast-reasoning
- Not supported by: grok-4-0709, grok-3 (these reason by default, reject the parameter)
- Maps from Shepherd's `--reasoning low|medium|high` flag

## Sampling Parameters
Only sends when explicitly configured (value >= 0) and `sampling = true`:
- `temperature`
- `top_p`
- `frequency_penalty`

## Provider Config
```json
{
    "api_key": "xai-...",
    "base_url": "https://api.x.ai/v1",
    "model": "",
    "name": "grok",
    "type": "grok"
}
```

## History
- v2.35.1: Created. Split from OpenAI backend to handle Grok-specific parameter support.
