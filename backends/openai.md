# OpenAI Backend

## Overview

The OpenAI backend provides an OpenAI-compatible API client for connecting to various LLM providers including OpenAI, Azure OpenAI, Anthropic (via proxy), local servers (vLLM, TRT-LLM, llama.cpp server), and other compatible APIs.

## Authentication

The backend supports two authentication methods:

### API Key (Standard OpenAI)
```json
{
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1"
}
```

### OAuth 2.0 (Azure OpenAI and Corporate Proxies)
```json
{
  "client_id": "your-client-id",
  "client_secret": "your-client-secret",
  "token_url": "https://login.microsoftonline.com/.../oauth2/v2.0/token",
  "token_scope": "https://cognitiveservices.azure.com/.default"
}
```

OAuth tokens are automatically acquired and refreshed (60-second buffer before expiry).

## Azure OpenAI

Azure OpenAI uses a different URL structure and requires additional configuration:

### Configuration
```json
{
  "type": "openai",
  "base_url": "https://your-resource.openai.azure.com",
  "deployment_name": "gpt-4",
  "api_version": "2024-06-01",
  "client_id": "...",
  "client_secret": "...",
  "token_url": "..."
}
```

### URL Format
Azure deployments use: `{base_url}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}`

### Parameter Differences
Azure OpenAI does **not** support `repetition_penalty`. The backend automatically excludes this parameter. Use `frequency_penalty` and `presence_penalty` instead.

## SSL Configuration

For corporate proxies with self-signed certificates:
```json
{
  "ssl_verify": false,
  "ca_bundle_path": "/path/to/custom-ca-bundle.pem"
}
```

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
- v2.5.2: Added OAuth 2.0 authentication, Azure OpenAI support (deployment-based URLs, api-version parameter), SSL configuration, removed repetition_penalty for Azure compatibility
