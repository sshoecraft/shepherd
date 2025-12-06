# API Backend Base Class

## Overview

The `ApiBackend` class is the base class for all API-based backends (OpenAI, Anthropic, Gemini, Ollama). It provides common functionality for HTTP-based API communication, token management, and OAuth authentication.

## Architecture

### Inheritance Hierarchy
```
Backend (base)
  └─ ApiBackend (API base)
       ├─ OpenAIBackend
       ├─ AnthropicBackend
       ├─ GeminiBackend
       └─ OllamaBackend
```

## Features

### HTTP Client Management
- Configurable timeouts (default: 120 seconds)
- SSL certificate verification control
- Custom CA bundle support
- Streaming and non-streaming requests

### OAuth 2.0 Support
The base class provides OAuth 2.0 client credentials flow:

**Configuration:**
- `client_id` - OAuth client identifier
- `client_secret` - OAuth client secret
- `token_url` - OAuth token endpoint
- `token_scope` - Optional scope parameter

**Token Management:**
- Automatic token acquisition on first request
- Token caching with expiry tracking
- Automatic refresh 60 seconds before expiry
- Thread-safe token storage

**Methods:**
- `acquire_oauth_token()` - Request new token from OAuth endpoint
- `ensure_valid_oauth_token()` - Check and refresh token if needed

### SSL Configuration
- `ssl_verify` - Enable/disable certificate verification (default: true)
- `ca_bundle_path` - Path to custom CA certificate bundle

### Token Counting
- Adaptive token estimation using EMA (Exponential Moving Average)
- Default: 2.5 chars/token (optimized for code-heavy content)
- Calibration mode for accurate token counting
- Delta-based tracking for efficient context management

### Common Parameters
All API backends support:
- `temperature` - Sampling temperature (0.0-2.0)
- `top_p` - Nucleus sampling threshold
- `top_k` - Top-k sampling (0=disabled)
- `frequency_penalty` - Reduce repetition (OpenAI format)
- `presence_penalty` - Encourage topic diversity (OpenAI format)
- `repeat_penalty` - Alternative repetition control (Ollama/TensorRT format)
- `stop_sequences` - Stop generation at specified strings

## Pure Virtual Methods

Concrete backends must implement:

- `parse_http_response()` - Parse HTTP response into unified Response structure
- `build_request()` - Build API request JSON from session and new message
- `build_request_from_session()` - Build request from complete session
- `parse_response()` - Extract generated text from API response
- `extract_tokens_to_evict()` - Parse context overflow errors
- `get_api_headers()` - Return authentication and content headers
- `get_api_endpoint()` - Return full endpoint URL
- `query_model_context_size()` - Query context size from API

## Streaming Support

### add_message_stream()

API backends can override `add_message_stream()` to provide token-by-token streaming:

```cpp
Response add_message_stream(Session& session, Message::Type type,
                           const std::string& content,
                           StreamCallback callback,
                           const std::string& tool_name = "",
                           const std::string& tool_id = "",
                           int prompt_tokens = 0,
                           int max_tokens = 0) override;
```

The callback is invoked for each token/chunk received:

```cpp
using StreamCallback = std::function<bool(const std::string& delta,
                                          const std::string& accumulated,
                                          const Response& partial)>;
```

Return `false` from the callback to stop generation early.

### Implementation by Backend

| Backend | Streaming Format | Notes |
|---------|-----------------|-------|
| Anthropic | SSE | Uses content_block_delta events |
| OpenAI | SSE | Uses choices[0].delta.content |
| Gemini | SSE | Uses streamGenerateContent?alt=sse endpoint |
| Ollama | NDJSON | Newline-delimited JSON with message.content |

### Default Behavior

The base `Backend::add_message_stream()` falls back to:
1. Call `add_message()` (non-streaming)
2. Invoke callback once with the complete response

This ensures compatibility with backends that don't implement native streaming.

## Configuration Loading

Backend-specific config is loaded via `parse_backend_config()`:
1. SSL settings (`ssl_verify`, `ca_bundle_path`)
2. OAuth credentials (`client_id`, `client_secret`, `token_url`, `token_scope`)
3. Sampling parameters (temperature, penalties, etc.)

## History

- v2.4.0: Initial ApiBackend abstraction
- v2.5.0: Added adaptive token counting with EMA
- v2.5.1: Improved error parsing for multiple API formats
- v2.5.2: Added OAuth 2.0 support and SSL configuration
- v2.6.1: Added add_message_stream() for Gemini and Ollama backends
