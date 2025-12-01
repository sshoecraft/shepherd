# Provider Module

## Overview

The provider module manages backend configurations for shepherd. Providers are stored as JSON files in `~/.config/shepherd/providers/` and allow users to configure multiple backend connections (local models, API services) with different settings.

## Architecture

### Classes

- **ProviderConfig** - Base class for all provider configurations
  - Common fields: type, name, model, priority, context_size, rate_limits, pricing
  - Pure virtual `to_json()` for serialization
  - Static `from_json()` factory method

- **LlamaProviderConfig** - llama.cpp backend configuration
  - model_path, tp, pp, gpu_layers, temperature, sampling params

- **TensorRTProviderConfig** - TensorRT-LLM backend configuration
  - model_path, tp, pp, gpu_id, temperature, sampling params

- **ApiProviderConfig** - OpenAI-compatible API configuration
  - api_key, base_url, temperature, max_tokens, stop_sequences

- **OllamaProviderConfig** - Ollama backend configuration
  - base_url, num_ctx, temperature, sampling params

- **Provider** - Manager class for loading/saving/selecting providers
  - `connect_provider()` - Connect to specific provider by name
  - `connect_next_provider()` - Try providers in priority order (respects auto_provider config)
  - `interactive_edit()` - CLI editor for provider configuration

## Config Fields

### Base Config (all providers)
- `type` - Backend type (llamacpp, tensorrt, openai, ollama)
- `name` - User-friendly identifier
- `model` - Model name or path
- `priority` - Selection priority (lower = higher priority)
- `context_size` - Context window size (0=auto)
- `rate_limits` - Token/request limits and cost caps
- `pricing` - Cost per million tokens (prompt/completion)

## Known Architectural Issues

### Config vs ProviderConfig Duplication

There is currently overlap between the global `Config` class and `ProviderConfig`:

**Global Config** contains:
- context_size, backend, model, api_key, streaming, etc.
- Command-line overrides are stored separately, not merged into Config

**ProviderConfig** contains:
- type, model, context_size, api_key, base_url, etc.
- Per-provider settings stored in JSON files

**Problems:**
1. Duplication of fields (context_size, model, api_key exist in both)
2. Unclear precedence - which value wins when both are set?
3. Command-line overrides don't update Config, used directly instead
4. `connect_provider()` takes context_size as a parameter AND reads it from provider config

**Current workaround:**
`connect_provider()` uses provider's context_size if non-zero, otherwise falls back to passed-in value.

**Proposed future design:**
1. Global Config should only hold shepherd-wide settings (streaming, truncate_limit, auto_provider, etc.)
2. Backend-specific settings (model, api_key, context_size, temperature, etc.) should only exist in ProviderConfig
3. Command-line overrides should either:
   - Update Config instance directly, OR
   - Be merged with provider settings when a provider is selected
4. Single source of truth for each setting

## History

- v2.4.0: Added provider abstraction and server implementation
- v2.4.1: Added auto_provider config, provider fallback on connection failure
- v2.5.0: Moved context_size to base ProviderConfig class
