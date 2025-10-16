# Shepherd

**Advanced Multi-Backend LLM System with Intelligent Memory Management**

Shepherd is a production-grade C++ LLM inference system supporting both local models (llama.cpp, TensorRT-LLM) and cloud APIs (OpenAI, Anthropic, Gemini, Grok, Ollama). It features sophisticated KV cache eviction policies, retrieval-augmented generation (RAG), hierarchical memory management, and comprehensive tool/function calling capabilities.

---

## Key Features

### 🔄 Multi-Backend Architecture
- **8 Backend Implementations**: Seamlessly switch between local inference and cloud APIs
  - **Local**: llama.cpp (CPU/GPU), TensorRT-LLM (NVIDIA optimized)
  - **Cloud**: OpenAI, Anthropic Claude, Google Gemini, xAI Grok, Ollama
- **Unified Interface**: Single API across all backends with automatic fallback
- **Dynamic Model Selection**: Runtime backend selection based on model availability

### 🧠 Intelligent KV Cache Eviction
- **Callback-Driven Eviction** (llama.cpp): Interrupt-based eviction triggered by memory pressure
- **Event-Driven Eviction** (TensorRT): Asynchronous monitoring with `KVCacheEventManager`
- **Smart Message Preservation**: Protects system prompts and current context
- **Automatic Archival**: Evicted messages stored in RAG database for retrieval
- **Position Shift Management**: Maintains KV cache consistency during eviction

### 📚 RAG System (Retrieval-Augmented Generation)
- **Memory-Mapped SQLite Storage**: Persistent conversation memory with mmap I/O optimization
- **FTS5 Full-Text Search**: Fast semantic search with BM25 ranking algorithm
- **Fact Storage**: Persistent key-value store for long-term knowledge
- **Automatic Archival**: Seamless integration with eviction system
- **Configurable Retention**: Database size limits and auto-pruning (default 10GB)

### 🛠️ Comprehensive Tools System
- **10+ Built-in Tools** across 6 categories:
  - **Filesystem**: read_file, write_file, list_directory, delete_file, file_exists
  - **Command Execution**: execute_command with timeout and signal handling
  - **HTTP Client**: GET, POST, PUT, DELETE with custom headers
  - **JSON Processing**: parse, validate, pretty-print, JSONPath extraction
  - **Memory Management**: search_memory, set_fact, get_fact, store_memory
  - **MCP Integration**: list_resources, read_resource, call_mcp_tool
- **Schema-Driven**: Structured parameter definitions with JSON schema support
- **Extensible**: Easy registration of custom tools via ToolRegistry

### 🔌 Model Context Protocol (MCP)
- **Full MCP Client/Server**: JSON-RPC 2.0 over stdio
- **External Tool Delegation**: Seamlessly integrate external MCP servers
- **Resource Access**: Read resources from MCP servers (files, databases, APIs)
- **Prompt Templates**: Reusable prompt templates from MCP servers

### 🌐 Server Mode (HTTP REST API)
- **OpenAI-Compatible Endpoints**: Remote access to Shepherd via REST API
  - `POST /v1/chat/completions` - Chat completions with streaming support
  - `GET /v1/models` - List available models
  - `GET /health` - Health check endpoint
- **Single-Session Architecture**: Stateful conversation with one client at a time
- **KV Cache Persistence**: Full conversation history maintained in KV cache
- **Tool Integration**: Client-side tool execution with OpenAI protocol
- **Remote Access**: Access local Shepherd instance from any OpenAI-compatible client

### 💾 Hierarchical Memory Management
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GPU VRAM      │    │   System RAM     │    │    Storage      │
│   (Tier 1)      │ ⟶  │   (Tier 2)       │ ⟶  │   (Tier 3)      │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • KV Cache      │    │ • Message Buffer │    │ • SQLite DB     │
│ • Active Tokens │    │ • Token Pools    │    │ • Conversation  │
│ • Model Weights │    │ • Spill Cache    │    │ • Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
     Hot (~ms)              Warm (~μs)            Cold (~10ms)
```

---

## Supported Backends

### Local Inference (Stateful)

#### **llama.cpp**
- **GPU Support**: CUDA with automatic layer offloading
- **Model Families**: Llama 3.x, Qwen 2.x, GLM-4, Mistral, Gemma, Phi, DeepSeek
- **Chat Templates**: Jinja2-based template rendering
- **Context Windows**: 8K - 256K+ tokens (model dependent)
- **KV Cache**: Callback-driven eviction with position tracking
- **Tokenization**: Accurate token counting via model vocabulary

#### **TensorRT-LLM**
- **Optimization**: NVIDIA TensorRT for maximum GPU performance
- **Multi-GPU**: MPI-based layer distribution across GPUs
- **Plugin Architecture**: Extensible with custom kernels
- **Context Windows**: 2K - 256K+ tokens
- **KV Cache**: Event-driven monitoring with asynchronous eviction
- **Production Ready**: Enterprise-grade inference with monitoring

### Cloud APIs (Stateless)

| Backend | Models | Context | Tools | Notes |
|---------|--------|---------|-------|-------|
| **OpenAI** | GPT-4, GPT-3.5-Turbo | 128K-200K | ✓ | Function calling |
| **Anthropic** | Claude 3/3.5 (Opus, Sonnet, Haiku) | 200K | ✓ | Separate system field |
| **Gemini** | Gemini Pro, 1.5 Pro/Flash, 2.0 | 32K-2M | ✓ | SentencePiece tokens |
| **Grok** | Grok-1, Grok-2 | 128K | ✓ | OpenAI-compatible |
| **Ollama** | Any Ollama model | 8K-128K | ✓ | Local/containerized |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  • main.cpp: Interactive loop, tool execution               │
│  • HTTP Server: REST API for external integrations          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────v──────────────────────────────────────┐
│                    Backend Manager                           │
│  • Unified interface across all backends                    │
│  • Automatic backend selection and fallback                 │
│  • Message formatting and tool conversion                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────v────┐  ┌──────v─────┐  ┌────v────────┐
│ LlamaCpp   │  │ TensorRT   │  │ API Backends│
│ Backend    │  │ Backend    │  │ (5 types)   │
└─────┬──────┘  └──────┬─────┘  └────┬────────┘
      │                │              │
┌─────v────────────────v──────────────v────────┐
│           Context Manager (Base)             │
│  • Message storage and token tracking        │
│  • Eviction calculation and archival         │
│  • Context utilization monitoring            │
└──────────────────┬───────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
┌───────v──┐  ┌────v────┐  ┌─v────────────┐
│ RAG      │  │ Tools   │  │ MCP Client   │
│ System   │  │ System  │  │              │
└──────────┘  └─────────┘  └──────────────┘
```

---

## KV Cache Eviction System

### Why Eviction Matters

Modern LLMs have limited GPU memory for storing conversation context (KV cache). A 70B model with 128K context window requires ~240GB of KV cache memory. Shepherd's eviction system enables **indefinite conversations** by intelligently managing this constraint.

### Eviction Strategies

#### **llama.cpp: Callback-Driven**
```cpp
// Triggered by llama.cpp when KV cache fills
ctx_params.kv_need_space_callback = [](uint32_t tokens_needed, void* user_data) {
    return backend->evict_to_free_space(tokens_needed);
};
```

**Process**:
1. Calculate oldest evictable messages (preserving system prompt and current message)
2. Map message indices to KV cache token positions
3. Remove token range from KV cache: `llama_memory_seq_rm()`
4. Shift remaining tokens to eliminate gaps: `llama_memory_seq_add()`
5. Archive evicted messages to RAG database
6. Insert eviction notice into context
7. Return new KV cache head position

#### **TensorRT: Event-Driven**
```cpp
// Background thread monitors KV cache events
void monitor_kv_events() {
    auto events = event_mgr->getLatestEvents(100ms);
    for (auto& event : events) {
        if (event.type == KVCacheRemovedData) {
            handle_eviction(event.block_hashes);
        }
    }
}
```

**Process**:
1. Asynchronous monitoring thread detects `KVCacheRemovedData` events
2. Estimate tokens removed (~64-128 tokens per block)
3. Calculate corresponding message range
4. Archive messages to RAG database
5. Update context state

### Protection Rules
- **System Message**: Never evicted (contains tools and instructions)
- **Current User Message**: Never evicted (needed for generation)
- **Recent Tool Calls**: Protected until response generated
- **Eviction Order**: Oldest-first (LRU) within evictable range

---

## Building Shepherd

### Prerequisites

**Required**:
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- SQLite3
- libcurl
- nlohmann/json

**Optional** (for local inference):
- llama.cpp (for CPU/GPU inference)
- TensorRT-LLM (for NVIDIA GPU optimization)
- CUDA Toolkit 11.8+ (for GPU support)

### Build Instructions

```bash
# Clone repository
git clone https://github.com/sshoecraft/shepherd.git
cd shepherd

# Build with CMake
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run
./shepherd
```

### CMake Options

```bash
# Build with llama.cpp support
cmake -DUSE_LLAMACPP=ON ..

# Build with TensorRT-LLM support
cmake -DUSE_TENSORRT=ON ..

# Build with all backends (default)
cmake -DUSE_LLAMACPP=ON -DUSE_TENSORRT=ON ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

---

## Configuration

### Backend Selection

Shepherd automatically detects available backends at runtime. Priority order:
1. **TensorRT-LLM** (if compiled and NVIDIA GPU detected)
2. **llama.cpp** (if compiled)
3. **API backends** (if API keys configured)

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export GROK_API_KEY="xai-..."

# Local Model Path (for llama.cpp/TensorRT)
export MODEL_PATH="/path/to/model.gguf"

# RAG Database
export RAG_DB_PATH="./shepherd_memory.db"
export RAG_MAX_SIZE="10737418240"  # 10GB

# Backend Override
export SHEPHERD_BACKEND="llamacpp"  # or "tensorrt", "openai", etc.
```

### Configuration File (shepherd.json)

```json
{
    "backend": "llamacpp",
    "model_path": "/models/llama-3.1-70b-instruct.gguf",
    "context_size": 131072,
    "gpu_layers": 48,
    "api_keys": {
        "openai": "sk-...",
        "anthropic": "sk-ant-..."
    },
    "rag": {
        "database_path": "./shepherd_memory.db",
        "max_size_bytes": 10737418240,
        "enable_archival": true
    },
    "tools": {
        "enabled": ["filesystem", "http", "memory", "command"],
        "disabled": ["command_execution"]
    },
    "mcp_servers": [
        {
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
        }
    ]
}
```

---

## Usage Examples

### Interactive Mode

```bash
$ ./shepherd /models/llama-3.1-70b-instruct.gguf

Shepherd LLM System v1.0
Backend: llama.cpp (CUDA)
Model: llama-3.1-70b-instruct
Context: 131072 tokens
Tools: 12 available

> What files are in the current directory?
[Tool Call: list_directory(path=".")]
[Tool Result: main.cpp, README.md, CMakeLists.txt, ...]

The current directory contains:
- main.cpp: Main application entry point
- README.md: Project documentation
- CMakeLists.txt: Build configuration
...

> Read the contents of README.md and summarize it
[Tool Call: read_file(path="README.md")]
[Tool Result: # Shepherd\n\nAdvanced Multi-Backend LLM System...]

This README describes Shepherd, a production-grade LLM inference system...
```

### Python API

```python
import requests

# Start conversation
response = requests.post("http://localhost:8080/v1/chat/completions", json={
    "model": "llama-3.1-70b-instruct",
    "messages": [
        {"role": "user", "content": "What's the weather like?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "http_get",
                "description": "Make HTTP GET request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    }
                }
            }
        }
    ]
})

print(response.json())
```

### Tool Execution

```bash
# Enable specific tools
./shepherd --enable-tools filesystem,http,memory

# Disable dangerous tools
./shepherd --disable-tools command_execution

# List available tools
./shepherd --list-tools
```

---

## Server Mode (HTTP REST API)

Shepherd can run as an HTTP server providing an OpenAI-compatible REST API for **remote access** to a local Shepherd instance. This enables you to access your local models from any OpenAI-compatible client, library, or tool.

**Important**: Server mode is **single-session** - designed for one user accessing their Shepherd instance remotely, not for multi-tenant production serving. For multi-user inference servers, consider vLLM or TGI.

**Example Use Cases**:
- Access your home server's GPU from your laptop
- Use OpenAI-compatible tools (like cursor.ai) with your local models
- Remote access to your Shepherd instance from anywhere
- Integration testing with OpenAI client libraries

**Not For**:
- Multi-user production deployments
- Serving multiple concurrent clients
- Building a SaaS product

### Starting the Server

```bash
# Start server with llamacpp backend
./shepherd --server --port 8080 --backend llamacpp --model /path/to/model.gguf

# Start with specific configuration
./shepherd --server --config server_config.json

# Server output:
Shepherd API Server starting...
Backend: llamacpp
Model: /models/qwen-3-30b.gguf
Listening on: http://0.0.0.0:8080

Endpoints:
  POST http://0.0.0.0:8080/v1/chat/completions
  GET  http://0.0.0.0:8080/v1/models
  GET  http://0.0.0.0:8080/health
```

### API Endpoints

#### POST `/v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request**:
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "List files in current directory"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "list_directory",
        "description": "List files in a directory",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string"}
          }
        }
      }
    }
  ],
  "temperature": 0.7,
  "max_tokens": 150
}
```

**Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "list_directory",
              "arguments": "{\"path\":\".\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

#### GET `/v1/models`

List available models.

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1234567890,
      "owned_by": "shepherd"
    }
  ]
}
```

#### GET `/health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "backend": "llamacpp",
  "model": "/models/qwen-3-30b.gguf"
}
```

### Session Management

Server mode implements a **single persistent session** that maintains the conversation state in the KV cache. This is designed for one user to access their Shepherd instance remotely.

#### How the Session Works

1. **Single Session**: The server maintains one active conversation session
2. **Message Accumulation**: Each request sends the **full conversation history** (OpenAI protocol)
3. **KV Cache Reuse**: Server maintains KV cache between requests for fast responses
4. **Prefix Matching**: Only new messages are processed; cached messages are skipped

**Note**: Unlike multi-tenant servers (vLLM, TGI), Shepherd's server mode is designed for **personal use** - think of it as remote access to your interactive Shepherd session, not as a production inference server.

**Example Session Flow**:

```python
import openai

client = openai.OpenAI(
    api_key="dummy",  # Not used
    base_url="http://localhost:8080/v1"
)

# Request 1: Send initial message
response1 = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ]
)
# Server: Creates session, caches system + user message
# KV Cache: [system_msg, user_msg, assistant_msg]

# Request 2: Send full history + new message
response2 = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"}
    ]
)
# Server: Detects prefix match, only processes new user message
# KV Cache: [system_msg, user_msg, assistant_msg, user_msg, assistant_msg]
```

#### Session Context Structure

The server maintains a single session with:

```cpp
struct SessionContext {
    std::vector<Message> messages;           // Full conversation history
    std::vector<Tool> tools;                 // Available tools from client
    size_t cached_message_count;             // Messages currently in KV cache
};
```

**Key Implementation Details**:

1. **Message Replacement** (not append): Server replaces `session.messages` with each request
   ```cpp
   session.messages.clear();  // Clear before adding new messages
   for (const auto& msg : request["messages"]) {
       session.messages.push_back(msg);
   }
   ```

2. **Prefix Matching**: Compares incoming messages with cached messages
   ```cpp
   // Find longest matching prefix
   size_t prefix_len = 0;
   for (size_t i = 0; i < min(cached_msgs, new_msgs); i++) {
       if (cached_messages[i] == new_messages[i])
           prefix_len++;
       else break;
   }
   ```

3. **Divergence Handling**: If conversation diverges, KV cache is cleared
   ```cpp
   if (prefix_len < cached_message_count) {
       LOG_WARN("Conversation diverged - clearing cache");
       clear_kv_cache();
   }
   ```

### Client Examples

#### Python (OpenAI Library)

```python
import openai

client = openai.OpenAI(
    api_key="dummy",
    base_url="http://localhost:8080/v1"
)

# Multi-turn conversation
messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})

    print(f"Assistant: {assistant_msg}")
```

#### curl

```bash
# Single request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# With tools
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "List files in /tmp"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "list_directory",
          "description": "List files",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string"}
            }
          }
        }
      }
    ]
  }'
```

### Configuration

Add to `shepherd.json`:

```json
{
    "server_mode": {
        "enabled": true,
        "host": "0.0.0.0",
        "port": 8080
    },
    "backend": "llamacpp",
    "model_path": "/models/model.gguf"
}
```

Or use command-line options:

```bash
./shepherd --server --host 0.0.0.0 --port 8080 --backend llamacpp --model /path/to/model.gguf
```

### Important Notes

- **Single-User Architecture**: Server mode is designed for **one user** accessing their Shepherd instance remotely
  - **Not multi-tenant**: Only one conversation at a time
  - **Not production-ready for multiple users**: Use vLLM, TGI, or similar for multi-user serving
  - **Use case**: Remote access to your local Shepherd (e.g., from laptop to home server)

- **Tools**: In server mode, tools are provided by the **client** in each request
  - Server does **NOT** execute tools (client-side execution only)
  - Server returns tool calls to client for execution
  - Client executes tools and sends results back in next request

- **KV Cache Persistence**: The session's KV cache persists across requests
  - Full conversation history maintained in memory
  - Prefix matching provides vLLM-like performance
  - Clear cache by restarting the server

- **Performance**: Server mode leverages KV cache for fast multi-turn conversations
  - First request: ~1.5s (full prompt processing)
  - Subsequent requests: ~200ms (only new tokens processed, 8-14x faster)

### KV Cache Persistence & Prefix Matching (vLLM-style)

Shepherd implements **automatic prefix caching** similar to vLLM's PagedAttention. When a new request arrives, the server:

1. **Compares** incoming messages with cached messages
2. **Detects** the longest matching prefix
3. **Reuses** KV cache for matching messages
4. **Only processes** new tokens after the prefix

**Example**:

```
Request 1: [system, user_1]
  → Server processes: system (4141 tokens) + user_1 (10 tokens)
  → KV cache: 4151 tokens
  → Time: ~1.5s (full processing)

Request 2: [system, user_1, assistant_1, user_2]
  → Server detects: system + user_1 match cached (4151 tokens)
  → Server processes ONLY: assistant_1 (27 tokens) + user_2 (8 tokens)
  → KV cache: 4151 + 35 = 4186 tokens
  → Time: ~200ms (90% faster!)

Request 3: [system, user_1, assistant_1, user_2, assistant_2, user_3]
  → Server detects: 4186 tokens already cached
  → Server processes ONLY: assistant_2 (105 tokens) + user_3 (12 tokens)
  → KV cache: 4186 + 117 = 4303 tokens
  → Time: ~180ms
```

**Logs showing prefix caching in action**:

```
[DEBUG] LlamaCpp generate_from_session called with 4 messages
[DEBUG] KV cache contains 2 messages, session has 4 messages
[DEBUG] Prefix match: 2 messages already cached
[DEBUG] Adding 2 new messages to KV cache
[DEBUG] Prompt already cached, skipping tokenization/decoding
[DEBUG] Generation limits: 150 max tokens (protected: system=4141 + user=7 + buffer=200)
[INFO ] Prompt processing: 13 tokens in 0.020000s (650 tokens/s)  ← Only new tokens!
```

**Performance Comparison**:

| Scenario | Without Prefix Caching | With Prefix Caching | Speedup |
|----------|------------------------|---------------------|---------|
| Turn 1 (4151 tokens) | 1.5s | 1.5s | 1x |
| Turn 2 (35 new tokens) | 1.6s | 0.2s | **8x faster** |
| Turn 3 (117 new tokens) | 1.7s | 0.18s | **9x faster** |
| Turn 10 (50 new tokens) | 2.1s | 0.15s | **14x faster** |

**Why This Matters**:

- **Multi-turn conversations** become nearly instant after the first turn
- **System prompts** (often 4K+ tokens) are processed once and cached
- **Long context** doesn't slow down subsequent requests
- **Scales** to very long conversations without performance degradation

**Implementation Details**:

The prefix matching algorithm in `generate_from_session()`:

```cpp
// Compare cached messages with incoming messages
size_t prefix_match_count = 0;
for (size_t i = 0; i < std::min(kv_cached_message_count_, session.messages.size()); i++) {
    if (kv_cached_messages_[i].role == session.messages[i].role &&
        kv_cached_messages_[i].content == session.messages[i].content) {
        prefix_match_count++;
    } else {
        // Conversation diverged - clear cache and start fresh
        LOG_WARN("Conversation diverged at message " + std::to_string(i));
        llama_kv_cache_clear(ctx);
        prefix_match_count = 0;
        break;
    }
}

LOG_DEBUG("Prefix match: " + std::to_string(prefix_match_count) + " messages already cached");

// Only add new messages after the prefix
for (size_t i = prefix_match_count; i < session.messages.size(); i++) {
    add_message_to_kv_cache(session.messages[i]);
}
```

This is functionally identical to vLLM's automatic prefix caching, providing the same performance benefits for multi-turn conversations.

---

## RAG System Usage

### Automatic Archival

When messages are evicted from KV cache, they're automatically archived:

```
[User]: What's the capital of France?
[Assistant]: Paris is the capital of France.

... 100K tokens later ...

[NOTICE: 2 messages evicted from context and archived to memory]
```

### Semantic Search

Retrieve relevant past conversations:

```python
# Search archived conversations
results = rag_manager.search("capital of France", max_results=5)

for result in results:
    print(f"Relevance: {result.score}")
    print(f"User: {result.user_message}")
    print(f"Assistant: {result.assistant_response}")
```

### Fact Storage

Store persistent knowledge across sessions:

```bash
> Remember that my favorite color is blue
[Tool Call: set_fact(key="favorite_color", value="blue")]
[Tool Result: {"success": true}]

I've stored that your favorite color is blue.

... new session ...

> What's my favorite color?
[Tool Call: get_fact(key="favorite_color")]
[Tool Result: {"value": "blue"}]

Your favorite color is blue!
```

### Direct Memory Storage

Store question/answer pairs directly to long-term memory:

```bash
> Store this information: "What is the project deadline?" Answer: "March 15, 2024"
[Tool Call: store_memory(question="What is the project deadline?", answer="March 15, 2024")]
[Tool Result: {"success": true}]

Stored to long-term memory.

... later in conversation or new session ...

> Search for information about the deadline
[Tool Call: search_memory(query="project deadline", max_results=3)]
[Tool Result: "Found 1 archived conversation(s):
Result 1 [Relevance: 0.95]:
User: What is the project deadline?
Assistant: March 15, 2024"]

The project deadline is March 15, 2024.
```

---

## Hardware Recommendations

### Optimal Configuration
- **GPUs**: 2x NVIDIA RTX 3090 (48GB VRAM) or better
- **RAM**: 128GB system RAM
- **Storage**: NVMe SSD (1TB+)
- **CPU**: 16+ cores for parallel processing

### Minimum Configuration
- **GPU**: NVIDIA GTX 1080 Ti (11GB VRAM) or better
- **RAM**: 32GB system RAM
- **Storage**: SATA SSD (500GB)
- **CPU**: 8 cores

### Cloud Deployment
- **AWS**: g5.12xlarge (4x A10G, 192GB RAM)
- **GCP**: a2-highgpu-4g (4x A100, 340GB RAM)
- **Azure**: Standard_NC24ads_A100_v4 (A100 80GB)

---

## Performance Characteristics

### Throughput (70B model, batch_size=1)

| Backend | Prompt Speed | Generation Speed | Latency |
|---------|--------------|------------------|---------|
| TensorRT-LLM | 8000 tok/s | 45 tok/s | ~50ms |
| llama.cpp (CUDA) | 1200 tok/s | 25 tok/s | ~80ms |
| llama.cpp (CPU) | 150 tok/s | 8 tok/s | ~200ms |

### Memory Usage (70B model)

| Configuration | VRAM | System RAM | Context Size |
|---------------|------|------------|--------------|
| Q4_K_M + 64K ctx | 38GB | 8GB | 65536 tokens |
| Q4_K_M + 128K ctx | 42GB | 12GB | 131072 tokens |
| Q8_0 + 64K ctx | 72GB | 16GB | 65536 tokens |

### Eviction Performance

- **Eviction latency**: <10ms (llama.cpp), <5ms (TensorRT)
- **Archive write**: ~2ms per conversation turn (SQLite)
- **Search latency**: ~15ms (FTS5 full-text search)
- **Context shift**: O(n) where n = remaining tokens

---

## Development

### Project Structure

```
shepherd/
├── backends/           # Backend implementations
│   ├── api_backend.{cpp,h}      # Base for API backends
│   ├── llamacpp.{cpp,h}         # llama.cpp backend
│   ├── tensorrt.{cpp,h}         # TensorRT-LLM backend
│   ├── openai.{cpp,h}           # OpenAI API
│   ├── anthropic.{cpp,h}        # Anthropic Claude
│   ├── gemini.{cpp,h}           # Google Gemini
│   ├── grok.{cpp,h}             # xAI Grok
│   └── ollama.{cpp,h}           # Ollama
├── tools/              # Tool implementations
│   ├── tool.{cpp,h}             # Base tool interface
│   ├── filesystem_tools.{cpp,h}
│   ├── http_tools.{cpp,h}
│   ├── command_tools.{cpp,h}
│   ├── json_tools.{cpp,h}
│   └── memory_tools.{cpp,h}
├── mcp/                # Model Context Protocol
│   ├── mcp_client.{cpp,h}
│   ├── mcp_server.{cpp,h}
│   └── mcp_manager.{cpp,h}
├── server/             # HTTP REST API Server
│   ├── api_server.py            # FastAPI wrapper (spawned by C++)
│   ├── requirements.txt         # Python dependencies
│   └── README.md                # Server documentation
├── server.{cpp,h}      # C++ HTTP server implementation
├── session_context.h   # Session state management
├── rag_system.{cpp,h}  # RAG implementation
├── memory_manager.{cpp,h}       # Memory management
├── context_manager.{cpp,h}      # Context handling
├── backend_manager.{cpp,h}      # Backend orchestration
├── model_config.h      # Model-specific configuration
├── tokenizer.{cpp,h}            # Tokenization
├── main.cpp            # Application entry (interactive & server modes)
└── CMakeLists.txt      # Build configuration
```

### Adding Custom Tools

```cpp
#include "tools/tool.h"

class MyCustomTool : public Tool {
public:
    std::string unsanitized_name() const override {
        return "my_custom_tool";
    }

    std::string description() const override {
        return "Does something amazing";
    }

    std::string parameters() const override {
        return R"({
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input text"}
            },
            "required": ["input"]
        })";
    }

    std::map<std::string, std::any> execute(
        const std::map<std::string, std::any>& args) override
    {
        std::string input = std::any_cast<std::string>(args.at("input"));
        // Do something with input
        return {{"result", "success"}};
    }
};

// Register tool
ToolRegistry::instance().register_tool(std::make_unique<MyCustomTool>());
```

### Adding Custom Backends

```cpp
#include "backend_interface.h"

class MyBackend : public Backend {
public:
    bool initialize(const std::string& model_path) override {
        // Initialize your backend
        return true;
    }

    std::string generate(const std::string& prompt,
                        const GenerationParams& params) override {
        // Generate response
        return response;
    }

    void add_message(const Message& message) override {
        // Add to context
    }

    // Implement other virtual methods...
};

// Register backend
BackendManager::register_backend("mybackend",
    std::make_unique<MyBackend>());
```

---

## Troubleshooting

### Out of Memory During Inference

**Solution**: Reduce context size or enable eviction
```bash
./shepherd --context-size 65536 --enable-eviction
```

### Slow Generation Speed

**Solution**: Increase GPU layers or use TensorRT
```bash
./shepherd --gpu-layers 48 --backend tensorrt
```

### Tool Execution Failures

**Solution**: Check tool permissions and enable verbose logging
```bash
./shepherd --verbose --log-level debug
```

### KV Cache Corruption

**Symptoms**: Repetitive or nonsensical output after eviction

**Solution**: Verify position shifting is working correctly. Check logs for:
```
[DEBUG] Evicting messages 2-5 (tokens 1024-4096)
[DEBUG] Shifting remaining tokens by -3072
[DEBUG] New KV cache size: 8192 tokens
```

### Infinite Tool Call Loops

**Symptoms**: Model repeatedly executes the same tool call in an infinite loop

**Root Cause**: Missing closing tags in KV cache after generation

This was a critical bug where assistant message closing tags (`<|im_end|>` for Qwen, `<|eot_id|>` for Llama) were never added to the KV cache after generation. This caused malformed context on the next generation cycle.

**Example of Bug**:
```
Iteration 1:
  <|im_start|>assistant
  {"id":"call_123","name":"list_directory"...}  ← MISSING <|im_end|>!
  <tool_response>...</tool_response><|im_end|>

Iteration 2:
  <|im_start|>assistant  ← DUPLICATE! Previous message was never closed
  {"id":"call_123","name":"list_directory"...}  ← SAME TOOL CALL AGAIN!
```

**Fix Applied** (backends/llamacpp.cpp:1054-1076):

The backend now adds the closing tag to the KV cache after generation completes:

```cpp
// CRITICAL FIX: After generation, add the closing tag to KV cache
if (!model_config_.assistant_end_tag.empty()) {
    // Tokenize and add the closing tag to KV cache
    const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model_));
    std::vector<llama_token> closing_tokens(16);
    int n_closing = llama_tokenize(vocab, model_config_.assistant_end_tag.c_str(),
                                    model_config_.assistant_end_tag.length(),
                                    closing_tokens.data(), closing_tokens.size(), false, true);

    if (n_closing > 0) {
        closing_tokens.resize(n_closing);
        llama_batch closing_batch = llama_batch_get_one(closing_tokens.data(), n_closing);
        if (llama_decode(ctx, closing_batch) != 0) {
            LOG_WARN("Failed to decode closing tag into KV cache");
        } else {
            LOG_DEBUG("Added closing tag to KV cache: " + model_config_.assistant_end_tag);
        }
    }
}
```

**Verification**: Check logs for confirmation:
```
[DEBUG] Added closing tag to KV cache: <|im_end|>
```

### Duplicate Messages in Server Mode

**Symptoms**: Same messages appear multiple times in KV cache during server mode conversations

**Root Cause**: Server was appending messages instead of replacing them

The OpenAI protocol sends the **full conversation history** with each request. The server must **replace** the session messages, not append them.

**Example of Bug**:
```
Request 1: [system, user] → session.messages has 2 messages
Request 2: [system, user, assistant, tool] → session.messages has 2 + 4 = 6 messages (duplicates!)
```

**Fix Applied** (server.cpp:160):

```cpp
// OpenAI protocol sends FULL conversation history each time, so REPLACE not append
session.messages.clear();  // Clear before adding new messages
for (const auto& msg : request["messages"]) {
    session.messages.push_back(m);
}
```

**Verification**: Check server logs show no duplicates:
```
[DEBUG] === MESSAGES IN KV CACHE ===
[DEBUG] [system] <|im_start|>system You are a helpful assistant...
[DEBUG] [user] list the files
[DEBUG] [assistant] {"id":"call_123"...
[DEBUG] [tool] Here are the files...
[DEBUG] === END KV CACHE ===
```

---

## Architecture Improvements

### Model-Agnostic Backend Design

Previously, backends contained hardcoded model family checks:

```cpp
// OLD CODE (BAD):
if (model_config_.family == ModelFamily::QWEN_2_X) {
    generation_prompt = "<|im_start|>assistant\n";
    closing_tag = "<|im_end|>\n";
} else if (model_config_.family == ModelFamily::LLAMA_3_X) {
    generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n";
    closing_tag = "<|eot_id|>";
}
```

This violated the separation of concerns - backends shouldn't know about specific model formats.

**New Design**: Model-specific tags are defined in `ModelConfig`:

```cpp
// model_config.h
struct ModelConfig {
    std::string assistant_start_tag;  // e.g., "<|im_start|>assistant\n"
    std::string assistant_end_tag;    // e.g., "<|im_end|>\n"
    // ...
};

// Populated by factory methods:
static ModelConfig create_qwen() {
    return ModelConfig{
        // ...
        .assistant_start_tag = "<|im_start|>assistant\n",
        .assistant_end_tag = "<|im_end|>\n"
    };
}
```

**Backend Usage** (now model-agnostic):

```cpp
// NEW CODE (GOOD):
std::string generation_prompt = model_config_.assistant_start_tag;
// ... generate ...
if (!model_config_.assistant_end_tag.empty()) {
    // Add closing tag
}
```

**Benefits**:
- Backends are model-agnostic
- Easy to add new model families without changing backend code
- Configuration-driven behavior
- Better separation of concerns

---

## Documentation

- **API Reference**: [docs/api.md](docs/api.md)
- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **Tool Development**: [docs/tools.md](docs/tools.md)
- **Backend Development**: [docs/backends.md](docs/backends.md)
- **MCP Integration**: [docs/mcp.md](docs/mcp.md)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional backend integrations
- New tool implementations
- Performance optimizations
- Documentation improvements
- Test coverage

---

## License

**PolyForm Shield License 1.0.0**

Copyright (C) 2024-2025 Stephen P. Shoecraft

This project is licensed under the PolyForm Shield License 1.0.0. You are free to:
- ✅ Use Shepherd for any purpose (personal, commercial, internal business use)
- ✅ Modify and create derivative works
- ✅ Distribute copies

**Restrictions**:
- ❌ Cannot sell Shepherd or derivatives as a standalone product
- ❌ Cannot offer Shepherd as a paid service (SaaS)
- ❌ Cannot create competing products using Shepherd

For the full license text, see [LICENSE](LICENSE) or visit:
https://polyformproject.org/licenses/shield/1.0.0/

**Commercial Licensing**: For use cases not covered by the PolyForm Shield License, please contact the author for alternative licensing options.

---

## Acknowledgments

- **llama.cpp**: Georgi Gerganov and contributors
- **TensorRT-LLM**: NVIDIA Corporation
- **Model Context Protocol**: Anthropic
- **SQLite**: D. Richard Hipp

---

## Contact

- **Issues**: https://github.com/sshoecraft/shepherd/issues
- **Discussions**: https://github.com/sshoecraft/shepherd/discussions
