# Anthropic Backend

## Overview

The `AnthropicBackend` class provides integration with Anthropic's Claude API. It supports both interactive CLI mode and API server mode with full SSE streaming.

## Features

- Full SSE streaming support for both `add_message()` and `generate_from_session()`
- Claude 3/3.5/4 model support (Opus, Sonnet, Haiku)
- Tool calling with streaming parameter parsing
- Automatic model configuration from Models database
- Token usage tracking via API response

## Streaming Support

### add_message() Streaming

The `add_message()` method provides streaming when `config->streaming` is enabled. It:
1. Adds `"stream": true` to the request
2. Uses `http_client->post_stream_cancellable()` with SSE parsing
3. Routes content through `process_output()` for filtering
4. Handles tool_use blocks with partial JSON accumulation
5. Updates session with messages and token counts

### generate_from_session() Streaming

The `generate_from_session()` method (used by API server mode) supports streaming when `config->streaming` is enabled. It:
1. Checks `config->streaming` - falls back to base class non-streaming if disabled
2. Builds request using `build_request_from_session()`
3. Adds `"stream": true` to the request
4. Uses SSE parsing to process events in real-time
5. Invokes callback for each content delta
6. Updates session token counts after completion

This enables proper SSE streaming when Shepherd is used as an API server proxy.

## SSE Event Types

The Anthropic API uses these event types during streaming:

| Event Type | Purpose |
|------------|---------|
| `message_start` | Contains initial usage (input_tokens) |
| `content_block_start` | Signals start of text or tool_use block |
| `content_block_delta` | Contains text deltas or partial_json for tools |
| `content_block_stop` | Signals end of a content block |
| `message_delta` | Contains stop_reason and final usage |
| `message_stop` | Signals message completion |
| `error` | Error during generation |

## Tool Calling

Tool calls are streamed as `tool_use` content blocks:
1. `content_block_start` with `type: "tool_use"` provides id and name
2. Multiple `content_block_delta` events stream `partial_json`
3. `content_block_stop` triggers JSON parsing and tool call emission

## Configuration

Provider configuration options:
- `api_key` - Anthropic API key (required)
- `model` - Model name (e.g., "claude-sonnet-4-5")
- `api_base` - Custom API endpoint (optional)
- `temperature`, `top_p`, `top_k` - Sampling parameters

## API Headers

Required headers for Anthropic API:
- `Content-Type: application/json`
- `x-api-key: <api_key>`
- `anthropic-version: 2023-06-01`
- Model-specific headers from `model_config.special_headers`

## History

- v2.4.0: Initial Anthropic backend
- v2.5.0: Added streaming support in add_message()
- v2.6.0: Added tool calling support
- v2.15.0: Added streaming generate_from_session() for API server mode
