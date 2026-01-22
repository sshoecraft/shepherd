# Shepherd

**Advanced Multi-Backend LLM System with Intelligent Memory Management**

Shepherd is a production-grade C++ LLM inference system supporting both local models (llama.cpp, TensorRT-LLM) and cloud APIs (OpenAI, Anthropic, Gemini, Ollama). It features KV cache eviction for indefinite conversations, retrieval-augmented generation (RAG), and comprehensive tool/function calling.

---

## Quick Start

**Local model:**
```bash
./shepherd -m /path/to/model.gguf
```

**Cloud provider:**
```bash
# Add a provider
shepherd provider add sonnet anthropic --model claude-sonnet-4 --api-key sk-ant-...

# Use it
./shepherd --provider sonnet
```

**Build from source:** See [BUILD.md](BUILD.md) for prerequisites and installation.

---

## Usage

### Interactive Mode

```bash
$ ./shepherd --provider mylocal

Shepherd v2.1.0
Provider: mylocal (llamacpp)
Model: qwen3-30b-a3b
Context: 40960 tokens
Tools: 18 available

> What files are in the current directory?
* list_directory(path=".")

The current directory contains:
- main.cpp: Application entry point
- README.md: Project documentation
- Makefile: Build configuration
...

> /provider use sonnet
Switched to provider: sonnet

> Explain this code
...
```

### Providers

Providers define backends you can switch between at runtime.

```bash
# List configured providers
shepherd provider list

# Add providers
shepherd provider add local llamacpp --model /models/qwen-72b.gguf
shepherd provider add sonnet anthropic --model claude-sonnet-4 --api-key sk-ant-...
shepherd provider add gpt openai --model gpt-4o --api-key sk-...

# Switch providers
shepherd provider use sonnet

# In interactive mode
> /provider use local
> /provider next
```

### Tools

Shepherd includes built-in tools across several categories:

| Category | Tools |
|----------|-------|
| **Filesystem** | read, write, list_directory, delete_file, file_exists |
| **Command** | shell (execute commands with timeout) |
| **HTTP** | http_get, http_post, http_put, http_delete |
| **JSON** | json_parse, json_validate, json_extract |
| **Memory** | search_memory, set_fact, get_fact, store_memory |
| **MCP** | list_resources, read_resource, call_mcp_tool |

```bash
# List tools
shepherd tools list

# Enable/disable specific tools
shepherd tools enable shell
shepherd tools disable shell

# Disable all tools
./shepherd --notools
```

---

## Configuration

### Config File

Configuration is stored at `~/.config/shepherd/config.json`:

```json
{
    "streaming": true,
    "thinking": false,
    "tui": true,
    "max_db_size": "10G",
    "memory_database": "~/.local/share/shepherd/memory.db"
}
```

```bash
# View configuration
shepherd config show

# Set values
shepherd config set streaming true
shepherd config set max_db_size 20G

# In interactive mode
> /config show
> /config set thinking true
```

### MCP Servers

Configure [Model Context Protocol](https://modelcontextprotocol.io/) servers for external tool integration:

```bash
# List MCP servers
shepherd mcp list

# Add an MCP server
shepherd mcp add mydb python /path/to/mcp_server.py -e DB_HOST=localhost

# Remove an MCP server
shepherd mcp remove mydb
```

### SMCP Servers (Secure Credentials)

SMCP passes credentials to MCP servers via stdin, never in environment variables or CLI args:

```bash
# Add SMCP server with credentials
shepherd smcp add database smcp-postgres --cred DB_URL=postgresql://user:pass@host/db
```

Credentials are sent via the [SMCP protocol](https://github.com/sshoecraft/smcp) handshake, never exposed in `/proc`, `ps`, or config files.

### Azure Key Vault

Load configuration from Azure Key Vault using Managed Identity:

```bash
./shepherd --config msi --kv my-vault-name
```

Store a secret named `shepherd-config` containing the unified JSON config. The VM's managed identity needs "Key Vault Secrets User" role.

### Environment Variables

```bash
SHEPHERD_INTERACTIVE=1    # Force interactive mode (useful in scripts/pipes)
NO_COLOR=1                # Disable colored output
```

---

## Server Modes

Shepherd can run as a server for remote access or persistent sessions.

### 🌐 API Server (OpenAI-Compatible)

Exposes an OpenAI-compatible REST API for remote access to your local Shepherd instance.

```bash
./shepherd --server --port 8000
```

**Use cases:**
- Access your home server's GPU from your laptop
- Use OpenAI-compatible tools with local models
- Integration with any OpenAI client library

**Endpoints:**
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `GET /v1/models` - List available models
- `GET /health` - Health check

**Authentication:**

Generate API keys for clients to authenticate against the server (OpenAI-compatible `Authorization: Bearer` header). See [docs/api_server.md](docs/api_server.md) for details.

```bash
./shepherd --server --auth-mode json
shepherd apikey add mykey    # Generates sk-shep-...
```

For full documentation, see [docs/api_server.md](docs/api_server.md).

### 🖥️ CLI Server (Persistent Session)

Runs a persistent AI session with server-side tool execution and multi-client access.

```bash
./shepherd --cliserver --port 8000
```

**Use cases:**
- 24/7 AI assistant with full tool access
- Query databases without exposing credentials to clients
- Multiple clients see the same session via SSE streaming

**Connect a client:**
```bash
./shepherd --backend cli --api-base http://server:8000
```

For full documentation, see [docs/cli_server.md](docs/cli_server.md).

### 🔗 Server Composability

Shepherd's architecture allows **any backend** with **any frontend**, and servers can be chained together.

**Key principle**: With API backends, each incoming connection creates a new backend connection - no session contention, fully scalable.

#### Example: API Proxy with Credential Isolation

Hide your Azure OpenAI credentials while adding tools and your own API keys:

```bash
# Shepherd connects to Azure OpenAI (credentials stay on server)
# Clients connect to Shepherd with your API keys
./shepherd --backend openai \
           --api-base https://mycompany.openai.azure.com/v1 \
           --api-key $AZURE_KEY \
           --server --port 8000 --auth-mode json --server-tools

# Generate keys for your clients
shepherd apikey add client1
shepherd apikey add client2
```

Clients get:
- Access to Azure OpenAI without knowing the Azure credentials
- Server-side tools (filesystem, shell, MCP servers)
- Your access control via Shepherd API keys

#### Example: Persistent Session on vLLM

Use vLLM's multi-user capabilities with a persistent CLI session:

```bash
# vLLM server running on port 5000 (handles multiple users efficiently)
# Shepherd CLI server on top for persistent session + tools
./shepherd --backend openai \
           --api-base http://localhost:5000/v1 \
           --cliserver --port 8000
```

Now you have:
- vLLM's PagedAttention for efficient multi-conversation handling
- Shepherd's persistent session (all clients see same conversation)
- Server-side tools executing locally

#### Example: Multi-Level Chaining

```bash
# Level 1: llamacpp backend
./shepherd --backend llamacpp -m /models/qwen-72b.gguf --server --port 5000

# Level 2: API server proxy (adds tools + API keys)
./shepherd --backend openai --api-base http://localhost:5000/v1 \
           --server --port 6000 --auth-mode json --server-tools

# Level 3: CLI server for persistent session
./shepherd --backend openai --api-base http://localhost:6000/v1 \
           --cliserver --port 7000
```

---

## Features

### 🔄 Multi-Backend Architecture

| Backend | Type | Models | Context | Tools |
|---------|------|--------|---------|-------|
| **llama.cpp** | Local | Llama, Qwen, Mistral, Gemma, etc. | 8K-256K | ✓ |
| **TensorRT-LLM** | Local | Same (NVIDIA optimized) | 2K-256K | ✓ |
| **OpenAI** | Cloud | GPT-5, GPT-4o, GPT-4 Turbo | 128K-200K | ✓ |
| **Anthropic** | Cloud | Claude Opus 4.5, Sonnet 4, Haiku | 200K | ✓ |
| **Gemini** | Cloud | Gemini 3, 2.5 Pro/Flash | 32K-2M | ✓ |
| **Azure OpenAI** | Cloud | GPT models via deployment | 128K-200K | ✓ |
| **Ollama** | Local/Cloud | Any Ollama model | 8K-128K | ✓ |

### 📚 RAG System

Evicted messages are automatically archived to a SQLite database with FTS5 full-text search:

```bash
> Remember that the project deadline is March 15
* set_fact(key="project_deadline", value="March 15")

# Later, or in a new session...
> What's the project deadline?
* get_fact(key="project_deadline")

The project deadline is March 15.
```

Search archived conversations:
```bash
> Search my memory for discussions about authentication
* search_memory(query="authentication")
```

### 🤝 Multi-Model Collaboration

When multiple providers are configured, Shepherd creates `ask_*` tools for cross-model consultation:

```bash
# Using local model, ask Claude for code review
> Read main.cpp and ask_sonnet to review it for bugs

* read(path="main.cpp")
* ask_sonnet(prompt="Review this code for bugs: ...")

Claude's analysis appears in your local model's context.
```

The current provider is excluded (you don't ask yourself). Switch providers and the tools update automatically.

### 🧠 KV Cache Management

Local backends (llama.cpp, TensorRT) use intelligent KV cache eviction for indefinite conversations:

- **Automatic eviction** when GPU memory fills
- **Oldest messages first** (LRU), protecting system prompt and current context
- **Automatic archival** to RAG database before eviction
- **Position shift management** maintains cache consistency

For implementation details, see [docs/llamacpp.md](docs/llamacpp.md).

### ⏰ Scheduling

Shepherd includes a cron-like scheduler that injects prompts into the session automatically. Works with CLI, TUI, and CLI server modes.

```bash
# Add a scheduled task (runs daily at 9am)
shepherd sched add morning-news "0 9 * * *" "Get me the top 5 tech news headlines"

# List scheduled tasks
shepherd sched list

# Enable/disable without removing
shepherd sched disable morning-news
shepherd sched enable morning-news
```

**24/7 Operation**: Run a CLI server and schedules execute automatically, even with no clients connected:

```bash
./shepherd --cliserver --port 8000

# Scheduled prompts run in the session:
# - "Check server disk usage" every hour
# - "Summarize overnight logs" at 6am
# - "Generate daily report" at 5pm
```

Clients connect to see results from scheduled tasks in the conversation history.

---

## Command Reference

### Subcommands

| Command | Description |
|---------|-------------|
| `shepherd provider <add\|list\|show\|remove\|use>` | Manage providers |
| `shepherd config <show\|set>` | View/modify configuration |
| `shepherd tools <list\|enable\|disable>` | Manage tools |
| `shepherd mcp <add\|remove\|list>` | Manage MCP servers |
| `shepherd smcp <add\|remove\|list>` | Manage SMCP servers |
| `shepherd sched <list\|add\|remove\|enable\|disable>` | Scheduled tasks |
| `shepherd apikey <add\|list\|remove>` | API key management |
| `shepherd edit-system` | Edit system prompt in $EDITOR |

### Common Flags

| Flag | Description |
|------|-------------|
| `-p, --provider NAME` | Use specific provider |
| `-m, --model PATH` | Model name or file |
| `--backend TYPE` | Backend: llamacpp, openai, anthropic, etc. |
| `--context-size N` | Context window size (0 = model default) |
| `--server` | Start API server mode |
| `--cliserver` | Start CLI server mode |
| `--port N` | Server port (default: 8000) |
| `--notools` | Disable all tools |
| `--nostream` | Disable streaming output |
| `--tui` / `--no-tui` | Enable/disable TUI mode |
| `--config msi --kv VAULT` | Load config from Azure Key Vault |

Run `shepherd --help` for the complete list.

---

## Hardware Requirements

### Minimum
- **GPU**: NVIDIA GTX 1080 Ti (11GB VRAM) or better
- **RAM**: 32GB system RAM
- **Storage**: SATA SSD (500GB)

### Recommended
- **GPU**: 2x NVIDIA RTX 3090 (48GB VRAM)
- **RAM**: 128GB system RAM
- **Storage**: NVMe SSD (1TB+)

### Cloud
- **AWS**: g5.12xlarge (4x A10G)
- **GCP**: a2-highgpu-4g (4x A100)
- **Azure**: Standard_NC24ads_A100_v4

---

## Performance

### Throughput (70B model, batch_size=1)

| Backend | Prompt Speed | Generation Speed | Latency |
|---------|--------------|------------------|---------|
| TensorRT-LLM | 8000 tok/s | 45 tok/s | ~50ms |
| llama.cpp (CUDA) | 1200 tok/s | 25 tok/s | ~80ms |
| llama.cpp (CPU) | 150 tok/s | 8 tok/s | ~200ms |

### Memory Usage (70B model)

| Configuration | VRAM | System RAM | Context |
|---------------|------|------------|---------|
| Q4_K_M + 64K ctx | 38GB | 8GB | 65536 |
| Q4_K_M + 128K ctx | 42GB | 12GB | 131072 |
| Q8_0 + 64K ctx | 72GB | 16GB | 65536 |

---

## Troubleshooting

### Out of Memory During Inference

Reduce context size:
```bash
./shepherd --context-size 65536
```

Or use a more aggressive quantization (Q4_K_M instead of Q8_0).

### Slow Generation Speed

Increase GPU layers or switch backends:
```bash
./shepherd --gpu-layers 48
```

### KV Cache Issues

If you see repetitive or nonsensical output, the KV cache may be corrupted. Restart Shepherd to clear the cache.

For debug builds, use `-d=3` for verbose KV cache logging.

---

## Development

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  CLI, TUI, API Server, CLI Server                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────v──────────────────────────────────┐
│                     Session + Provider                       │
│  Message routing, provider switching, tool execution        │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────v────┐      ┌──────v─────┐     ┌─────v─────────┐
│  LlamaCpp  │      │  TensorRT  │     │ API Backends  │
│  Backend   │      │  Backend   │     │ (5 types)     │
└────────────┘      └────────────┘     └───────────────┘
```

For detailed architecture, see [docs/architecture.md](docs/architecture.md).

### Project Structure

```
shepherd/
├── main.cpp              # Application entry point
├── frontend.cpp/h        # Frontend abstraction
├── backend.cpp/h         # Backend base class
├── session.cpp/h         # Session management
├── provider.cpp/h        # Provider management
├── config.cpp/h          # Configuration
├── rag.cpp/h             # RAG system
├── server.cpp/h          # HTTP server base
│
├── backends/
│   ├── llamacpp.cpp/h    # llama.cpp backend
│   ├── tensorrt.cpp/h    # TensorRT-LLM backend
│   ├── openai.cpp/h      # OpenAI API
│   ├── anthropic.cpp/h   # Anthropic Claude
│   ├── gemini.cpp/h      # Google Gemini
│   ├── ollama.cpp/h      # Ollama
│   ├── api.cpp/h         # Base for API backends
│   └── factory.cpp/h     # Backend factory
│
├── frontends/
│   ├── cli.cpp/h         # CLI frontend
│   ├── tui.cpp/h         # TUI frontend
│   ├── api_server.cpp/h  # API server
│   └── cli_server.cpp/h  # CLI server
│
├── tools/                # Tool implementations
├── mcp/                  # MCP client/server
└── Makefile              # Build system
```

### Extending Shepherd

- **Adding backends**: See [docs/backends.md](docs/backends.md)
- **Adding tools**: See [docs/tools.md](docs/tools.md) (if exists) or `tools/tool.h`

---

## Contributing

Contributions welcome! Areas of interest:
- Additional backend integrations
- New tool implementations
- Performance optimizations
- Documentation improvements

---

## Testing

```bash
# Build with tests enabled
echo "TESTS=ON" >> ~/.shepherd_opts
make

# Run tests
cd build && make test_unit test_tools
./tests/test_unit
./tests/test_tools
```

See [docs/testing.md](docs/testing.md) for the full test plan.

---

## License

**PolyForm Shield License 1.0.0**

- ✅ Use for any purpose (personal, commercial, internal)
- ✅ Modify and create derivative works
- ✅ Distribute copies
- ❌ Sell Shepherd as a standalone product
- ❌ Offer Shepherd as a paid service (SaaS)
- ❌ Create competing products

See [LICENSE](LICENSE) for full text.

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
