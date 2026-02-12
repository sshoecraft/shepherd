# Shepherd CLI Client Protocol

This document describes how to implement a client that connects to Shepherd's CLI Server mode. The reference implementation is `backends/cli_client.cpp`.

## Overview

Shepherd CLI Server exposes an HTTP API that allows external clients (web UIs, IDEs, custom tools) to interact with Shepherd. The server handles:

- Session management (messages, token tracking)
- Context window management (automatic eviction when full)
- Tool execution (MCP servers, built-in tools)
- Streaming responses via Server-Sent Events (SSE)

Clients are simple - they send prompts and receive events. The server owns all state.

## Architecture

```
┌─────────────────┐         ┌─────────────────────────────────────┐
│     Client      │         │       Shepherd CLI Server           │
│  (Open WebUI,   │         │                                     │
│   IDE plugin,   │  HTTP   │  ┌─────────┐    ┌─────────────────┐ │
│   custom app)   │◄───────►│  │ Session │    │    Backend      │ │
│                 │         │  │         │◄──►│ (llama.cpp,     │ │
│                 │   SSE   │  │ Messages│    │  vLLM, API,     │ │
│                 │◄────────│  │ Tokens  │    │  TensorRT)      │ │
└─────────────────┘         │  │ Tools   │    └─────────────────┘ │
                            │  └─────────┘                        │
                            └─────────────────────────────────────┘
```

## Endpoints

### GET /session

Fetch current session state. Call this on connect to sync with server.

**Request:**
```http
GET /session HTTP/1.1
Authorization: Bearer <api_key>  (optional)
```

**Response:**
```json
{
  "success": true,
  "context_size": 32768,
  "model": "llama-3.1-8b",
  "total_tokens": 4521,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!",
      "tokens": 12
    },
    {
      "role": "assistant",
      "content": "Hi there! How can I help you today?",
      "tokens": 15,
      "tool_calls": []  // OpenAI format if present
    },
    {
      "role": "tool",
      "content": "{\"result\": \"success\"}",
      "tool_call_id": "call_abc123"
    }
  ],
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web",
      "parameters": { ... }
    }
  ]
}
```

### POST /request

Send a user message and trigger generation.

**Request:**
```http
POST /request HTTP/1.1
Content-Type: application/json
Authorization: Bearer <api_key>  (optional)

{
  "prompt": "What is the capital of France?",
  "stream": false,
  "max_tokens": 4096
}
```

**Parameters:**
- `prompt` (required): The user message to process
- `stream` (optional, default: false): If true, returns SSE stream; if false, returns JSON
- `async` (optional, default: false): If true, queues request and returns immediately
- `max_tokens` (optional, default: 0): Maximum generation tokens
  - `0` = auto (server calculates based on available context)
  - `-1` = maximum available (use all remaining context)
  - `>0` = explicit limit (cap generation at this many tokens)

**Response (non-streaming):**
```json
{
  "success": true,
  "response": "The capital of France is Paris."
}
```

**Note:** Even with `stream: false`, real-time output is delivered via the `/updates` SSE connection. The POST response just confirms completion.

### GET /updates

Server-Sent Events stream for real-time updates. Connect once and keep open.

**Request:**
```http
GET /updates HTTP/1.1
Accept: text/event-stream
Authorization: Bearer <api_key>  (optional)
```

**Response:** SSE stream (see Events section below)

### POST /clear

Clear the conversation (reset session).

**Request:**
```http
POST /clear HTTP/1.1
Authorization: Bearer <api_key>  (optional)
```

**Response:**
```json
{
  "success": true,
  "message": "Conversation cleared"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### GET /status

Detailed server status.

**Response:**
```json
{
  "status": "ok",
  "model": "llama-3.1-8b",
  "context_size": 32768,
  "total_tokens": 4521,
  "processing": false
}
```

## SSE Events

All events are JSON objects with `type` and `data` fields:

```
data: {"type": "<event_type>", "data": { ... }}
```

### delta

Streaming text chunk from model generation.

```json
{"type": "delta", "data": {"delta": "The capital"}}
{"type": "delta", "data": {"delta": " of France"}}
{"type": "delta", "data": {"delta": " is Paris."}}
```

**Client action:** Append `delta` to current response content.

### codeblock

Code block content (when model outputs code).

```json
{"type": "codeblock", "data": {"content": "def hello():\n    print('Hello')"}}
```

**Client action:** Append to content with code formatting.

### message_added

A complete message was added to the session.

```json
{"type": "message_added", "data": {
  "role": "user",
  "content": "What is 2+2?",
  "tokens": 8
}}
```

```json
{"type": "message_added", "data": {
  "role": "assistant",
  "content": "2+2 equals 4.",
  "tokens": 12
}}
```

**Client action:** Add message to history with token count. Use for syncing state.

### tool_call

Model is calling a tool. Server will execute it.

```json
{"type": "tool_call", "data": {
  "tool_call": "web_search",
  "parameters": {"query": "weather in Paris"},
  "tool_call_id": "call_abc123"
}}
```

**Client action:** Display tool call UI (name, parameters). Do NOT execute - server handles it.

### tool_result

Result of tool execution.

```json
{"type": "tool_result", "data": {
  "tool_name": "web_search",
  "success": true
}}
```

```json
{"type": "tool_result", "data": {
  "tool_name": "web_search",
  "success": false,
  "error": "Network timeout"
}}
```

**Client action:** Display result status. Generation will continue after tool results.

### response_complete

Model finished generating response.

```json
{"type": "response_complete", "data": {
  "response": "The capital of France is Paris."
}}
```

**Client action:** Mark generation complete. Save chat if needed.

### error

An error occurred.

```json
{"type": "error", "data": {
  "error": "Model failed to generate response"
}}
```

**Client action:** Display error to user.

### eviction

Messages were evicted to free context space.

```json
{"type": "eviction", "data": {
  "messages_evicted": 3,
  "tokens_freed": 2048,
  "total_tokens": 8192
}}
```

**Client action:**
- Display notification (e.g., "3 older messages removed to fit context")
- Remove evicted messages from UI (oldest N messages)
- Update token count display

### Stream End

The stream ends with a done marker:

```
data: {"done": true}
```

## Client Implementation Pattern

Based on `backends/cli_client.cpp`:

### 1. Initialization

```python
class ShepherdClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.context_size = 0
        self.model_name = ""
        self.total_tokens = 0

        # Fetch initial session state
        session = self.get_session()
        self.context_size = session.get("context_size", 0)
        self.model_name = session.get("model", "")
        self.total_tokens = session.get("total_tokens", 0)

        # Display existing messages
        for msg in session.get("messages", []):
            self.display_message(msg)

        # Start SSE listener
        self.start_sse_listener()
```

### 2. SSE Listener

Run in background thread/task. Handles all real-time events.

```python
async def sse_listener(self):
    while self.running:
        try:
            async with self.http_client.stream("GET", f"{self.base_url}/updates") as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        await self.handle_event(data)
        except ConnectionError:
            # Reconnect after delay
            await asyncio.sleep(2)

async def handle_event(self, event: dict):
    event_type = event.get("type", "")
    event_data = event.get("data", {})

    if event_type == "delta":
        self.on_delta(event_data["delta"])
    elif event_type == "message_added":
        self.on_message_added(event_data)
    elif event_type == "tool_call":
        self.on_tool_call(event_data)
    elif event_type == "tool_result":
        self.on_tool_result(event_data)
    elif event_type == "response_complete":
        self.on_response_complete(event_data)
    elif event_type == "error":
        self.on_error(event_data["error"])
    elif event_type == "eviction":
        self.on_eviction(event_data)
    elif event_type == "codeblock":
        self.on_codeblock(event_data["content"])
```

### 3. Sending Messages

```python
async def send_message(self, prompt: str, max_tokens: int = 0) -> dict:
    request = {"prompt": prompt, "stream": False}
    if max_tokens != 0:
        request["max_tokens"] = max_tokens
    response = await self.http_client.post(
        f"{self.base_url}/request",
        json=request,
        headers=self.get_headers()
    )
    return response.json()
```

### 4. Event Handlers

```python
def on_delta(self, text: str):
    """Streaming text chunk - append to current response"""
    self.current_response += text
    self.update_display(text)

def on_message_added(self, data: dict):
    """Complete message added to session"""
    role = data["role"]
    content = data["content"]
    tokens = data.get("tokens", 0)
    self.messages.append({"role": role, "content": content, "tokens": tokens})
    self.total_tokens = sum(m.get("tokens", 0) for m in self.messages)

def on_tool_call(self, data: dict):
    """Display tool call - server executes it"""
    tool_name = data["tool_call"]
    params = data.get("parameters", {})
    self.display_tool_call(tool_name, params)

def on_tool_result(self, data: dict):
    """Display tool result"""
    success = data["success"]
    error = data.get("error", "")
    self.display_tool_result(data["tool_name"], success, error)

def on_response_complete(self, data: dict):
    """Generation finished"""
    self.generating = False
    self.save_chat()

def on_error(self, error: str):
    """Display error"""
    self.display_error(error)

def on_eviction(self, data: dict):
    """Messages were evicted"""
    count = data["messages_evicted"]
    tokens = data["tokens_freed"]
    self.display_notification(f"{count} older messages removed ({tokens} tokens freed)")
    # Remove oldest messages from local state
    self.messages = self.messages[count:]

def on_codeblock(self, content: str):
    """Code block content"""
    self.current_response += content
    self.update_display_code(content)
```

## Authentication

If the server requires authentication, include the API key in requests:

```http
Authorization: Bearer <api_key>
```

Or configure in the client:

```python
def get_headers(self):
    headers = {"Content-Type": "application/json"}
    if self.api_key:
        headers["Authorization"] = f"Bearer {self.api_key}"
    return headers
```

## Session Ownership

**Key principle:** The server owns the session state. Clients sync from the server.

- Token counts come from the server (accurate, not estimated)
- Message history is authoritative on the server
- Evictions happen server-side; clients just update their display
- Tool execution happens server-side; clients just display status

This means clients can be simple and stateless - they just render what the server tells them.

## Reconnection

The SSE connection may drop. Clients should:

1. Detect disconnection
2. Wait briefly (e.g., 2 seconds)
3. Reconnect to `/updates`
4. Optionally call `/session` to resync state

```python
async def sse_listener(self):
    while self.running:
        try:
            async with self.connect_sse() as stream:
                async for event in stream:
                    await self.handle_event(event)
        except (ConnectionError, TimeoutError):
            if self.running:
                await asyncio.sleep(2)  # Wait before reconnect
```

## Example: Minimal Client

```python
import asyncio
import httpx
import json

class MinimalShepherdClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.running = True

    async def run(self):
        # Start SSE listener in background
        asyncio.create_task(self.sse_listener())

        # Interactive loop
        async with httpx.AsyncClient() as client:
            while True:
                prompt = input("> ")
                if prompt.lower() == "quit":
                    break
                await client.post(
                    f"{self.base_url}/request",
                    json={"prompt": prompt, "stream": False}
                )

        self.running = False

    async def sse_listener(self):
        async with httpx.AsyncClient() as client:
            while self.running:
                try:
                    async with client.stream("GET", f"{self.base_url}/updates") as r:
                        async for line in r.aiter_lines():
                            if line.startswith("data: "):
                                event = json.loads(line[6:])
                                self.handle_event(event)
                except Exception:
                    await asyncio.sleep(2)

    def handle_event(self, event: dict):
        t = event.get("type", "")
        d = event.get("data", {})

        if t == "delta":
            print(d.get("delta", ""), end="", flush=True)
        elif t == "response_complete":
            print()  # Newline after response
        elif t == "tool_call":
            print(f"\n[Tool: {d.get('tool_call')}]")
        elif t == "error":
            print(f"\nError: {d.get('error')}")
        elif t == "eviction":
            print(f"\n[{d.get('messages_evicted')} messages evicted]")

if __name__ == "__main__":
    client = MinimalShepherdClient("http://localhost:8080")
    asyncio.run(client.run())
```

## Reference Implementation

See `backends/cli_client.cpp` for the complete C++ reference implementation used by Shepherd's CLI frontend when connecting to a remote CLI server.

Key sections:
- Constructor: Initialization and session sync
- `sse_listener_thread`: SSE event handling
- `send_request`: Sending messages with optional max_tokens
- `generate_from_session`: Passes max_tokens through to server
