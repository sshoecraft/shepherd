# Shepherd

**Multi-Backend LLM Inference Server and Interactive Agent**

Shepherd is a C++ LLM system supporting local models (llama.cpp, TensorRT-LLM) and cloud APIs (OpenAI, Anthropic, Gemini, Grok, Ollama). It provides multiple frontends (CLI, TUI, OpenAI-compatible API server, CLI server, JSON line-protocol), automatic KV cache eviction for indefinite conversations, retrieval-augmented generation (RAG) with SQLite or PostgreSQL, tool/function calling, MCP integration, multi-model collaboration, and background memory extraction.

---

## Quick Start

```bash
git clone --recursive https://github.com/sshoecraft/shepherd.git
cd shepherd
make
```

The Makefile checks for missing dependencies and tells you exactly what to install.

**Add a cloud provider and start using it:**
```bash
# OpenAI
shepherd provider add gpt openai --model gpt-4o --api-key sk-...

# Anthropic
shepherd provider add sonnet anthropic --model claude-sonnet-4 --api-key sk-ant-...

# Start chatting
./shepherd --provider sonnet
```

**Local model (requires NVIDIA GPU + CUDA):**
```bash
./shepherd -m /path/to/model.gguf
```

See [BUILD.md](BUILD.md) for GPU setup and advanced build options.

---

## Usage

### Interactive Mode

```bash
$ ./shepherd --provider mylocal

Shepherd v2.39.5
Provider: mylocal (llamacpp)
Model: qwen3-30b-a3b
Context: 40960 tokens
Tools: 50+ available

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

# Add providers (action-first for creating new)
shepherd provider add local --type llamacpp --model /models/qwen-72b.gguf
shepherd provider add sonnet --type anthropic --model claude-sonnet-4 --api-key sk-ant-...
shepherd provider add gpt --type openai --model gpt-4o --api-key sk-...

# View/modify providers (name-first pattern)
shepherd provider sonnet show        # Show details
shepherd provider sonnet set model claude-sonnet-4-20250514  # Modify setting
shepherd provider sonnet use         # Switch to this provider
shepherd provider sonnet             # Show help for this provider

# In interactive mode
> /provider local use
> /provider next
```

### Tools

Shepherd includes built-in tools across several categories:

| Category | Tools |
|----------|-------|
| **Core** | bash, glob, grep, edit, web_fetch, web_search, todo_write, task, get_time, get_date |
| **Filesystem** | read, write, list_directory, delete_file, file_exists |
| **Command** | execute_command, get_environment_variable, list_processes |
| **HTTP** | http_get, http_post, http_put, http_delete |
| **JSON** | json_parse, json_serialize, json_extract |
| **Memory** | search_memory, set_fact, get_fact, clear_fact, store_memory, clear_memory |
| **Scheduler** | list_schedules, add_schedule, remove_schedule, enable_schedule, disable_schedule, get_schedule |
| **MCP** | list_mcp_resources, read_mcp_resource, plus dynamic `server:tool` from configured MCP servers |
| **API Provider** | `ask_<provider>` tools for cross-model consultation (auto-generated from configured providers) |
| **Remote** | Remote tool proxy for distributed tool execution |

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

Configuration is stored at `~/.config/shepherd/config.json` (XDG-compliant):

```json
{
    "streaming": true,
    "thinking": false,
    "reasoning": "off",
    "tui": true,
    "stats": false,
    "auto_provider": false,
    "warmup": true,
    "max_tokens": 0,
    "memory_database": "~/.local/share/shepherd/memory.db",
    "max_db_size": "10G",
    "rag_context_injection": true,
    "rag_relevance_threshold": 0.3,
    "rag_max_results": 5,
    "memory_extraction": false,
    "memory_extraction_model": "",
    "memory_extraction_endpoint": "",
    "user_id": "",
    "web_search_provider": "",
    "auth_mode": "none",
    "server_tools": false
}
```

```bash
# View configuration
shepherd config show

# Set values (key-first shortcut)
shepherd config streaming true       # Set streaming to true
shepherd config max_db_size 20G      # Set max_db_size

# Or use explicit set
shepherd config set streaming true

# View single value
shepherd config streaming            # Shows current streaming value

# In interactive mode
> /config show
> /config streaming true
```

### MCP Servers

Configure [Model Context Protocol](https://modelcontextprotocol.io/) servers for external tool integration:

```bash
# List MCP servers
shepherd mcp list

# Add an MCP server (action-first for creating new)
shepherd mcp add mydb python /path/to/mcp_server.py -e DB_HOST=localhost

# View/modify servers (name-first pattern)
shepherd mcp mydb show               # Show server details
shepherd mcp mydb test               # Test connection
shepherd mcp mydb remove             # Remove server
shepherd mcp mydb                    # Show help for this server
```

### SMCP Servers (Secure Credentials)

SMCP passes credentials to MCP servers via stdin, never in environment variables or CLI args:

```bash
# Add SMCP server with credentials
shepherd smcp add database smcp-postgres --cred DB_URL=postgresql://user:pass@host/db
```

Credentials are sent via the [SMCP protocol](https://github.com/sshoecraft/smcp) handshake, never exposed in `/proc`, `ps`, or config files.

### External Configuration Sources

**Azure Key Vault** -- Load configuration from Azure Key Vault using Managed Identity:

```bash
./shepherd --config msi --kv my-vault-name
```

Store a secret named `shepherd-config` containing the unified JSON config. The VM's managed identity needs "Key Vault Secrets User" role.

**HashiCorp Vault** -- Load configuration from HashiCorp Vault using a pre-injected token:

```bash
./shepherd --config vault --kv https://vault.example.com
```

Reads the token from `/vault/secrets/token` (Vault Agent Injector) and fetches config from `shepherd/config` (KV v2).

Configuration loaded from either vault is **read-only** -- settings cannot be modified at runtime.

### Environment Variables

```bash
SHEPHERD_INTERACTIVE=1    # Force interactive mode (useful in scripts/pipes)
NO_COLOR=1                # Disable colored output
```

---

## Server Modes

Shepherd can run as a server for remote access or persistent sessions.

### API Server (OpenAI-Compatible)

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
- `GET /v1/models` - List available models (includes provider info, version, capabilities)
- `GET /health` - Health check

**Authentication:**

Generate API keys for clients to authenticate against the server (OpenAI-compatible `Authorization: Bearer` header). See [docs/api_server.md](docs/api_server.md) for details.

```bash
./shepherd --server --auth-mode json
shepherd apikey create mykey    # Generates sk-shep-...
```

**Server-side tool execution:**

```bash
# Tools execute on the server, results returned to client
./shepherd --server --use-tools

# Expose /v1/tools endpoint for tool discovery
./shepherd --server --server-tools

# Control whether tool call details are streamed to clients
./shepherd --server --use-tools --show-tool-calls true
```

For full documentation, see [docs/api_server.md](docs/api_server.md).

### CLI Server (Persistent Session)

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

### JSON Frontend (Machine Integration)

A JSON line-protocol frontend for pipe-based machine-to-machine communication (chatroom adapters, orchestration scripts, test harnesses).

```bash
# Interactive pipe mode
shepherd --json -p provider_name

# Single query mode
shepherd --json --prompt "hello" -p provider_name

# Piped input
echo '{"type":"user","content":"hello"}' | shepherd --json -p provider_name
```

**Input** (stdin, one JSON object per line):
```json
{"type": "user", "content": "your message here"}
```

**Output** (stdout, one JSON object per line):

| Type | Description | Fields |
|------|-------------|--------|
| `text` | Assistant response chunk | `content` |
| `thinking` | Reasoning/thinking chunk | `content` |
| `tool_use` | Tool call initiated | `name`, `params`, `id` |
| `tool_result` | Tool execution result | `name`, `id`, `success`, `summary` |
| `end_turn` | Turn complete | `turns`, `total_tokens`, `cost_usd` |
| `error` | Error occurred | `message` |
| `system` | System message | `content` |

Tools execute locally (same as CLI/TUI). No threads, no terminal handling, no HTTP -- the simplest frontend.

### Server Composability

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
shepherd apikey create client1
shepherd apikey create client2
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

### Multi-Backend Architecture

| Backend | Type | Description | Tools |
|---------|------|-------------|-------|
| **llama.cpp** | Local | Llama, Qwen, Mistral, Gemma, and other GGUF models | Yes |
| **TensorRT-LLM** | Local | NVIDIA-optimized inference for supported models | Yes |
| **OpenAI** | Cloud | GPT-4o, GPT-4 Turbo, o1, o3 (also Azure OpenAI deployments) | Yes |
| **Anthropic** | Cloud | Claude Opus, Sonnet, Haiku | Yes |
| **Gemini** | Cloud | Gemini 2.5 Pro/Flash | Yes |
| **Grok** | Cloud | xAI Grok models (OpenAI-compatible protocol) | Yes |
| **Ollama** | Local/Cloud | Any model available in Ollama | Yes |
| **CLI Client** | Remote | Connects to a remote Shepherd CLI server | Yes |

### RAG System

Evicted messages are automatically archived to a database with full-text search. Supports two backends:

- **SQLite** (default): FTS5 full-text search with BM25 ranking + time-based recency scoring
- **PostgreSQL** (optional): `tsvector`/`tsquery` with GIN index + recency scoring

```bash
# SQLite (default)
shepherd config memory_database ~/.local/share/shepherd/memory.db

# PostgreSQL
shepherd config memory_database postgresql://user:pass@host/shepherd
```

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

**Memory extraction**: An optional background thread that automatically extracts facts and context summaries from conversations using a separate LLM API call. Enable with `memory_extraction: true` in config and configure `memory_extraction_endpoint` and `memory_extraction_api_key`.

**Multi-tenant isolation**: All RAG operations are scoped by `user_id`. Set a global `user_id` in config to share memory across platforms, or leave empty for automatic per-client isolation.

### Multi-Model Collaboration

When multiple providers are configured, Shepherd creates `ask_*` tools for cross-model consultation:

```bash
# Using local model, ask Claude for code review
> ask_sonnet to read main.cpp and suggest improvements

* ask_sonnet(prompt="read main.cpp and suggest improvements")
  → Sonnet calls read(path="main.cpp")
  → Sonnet analyzes and responds

Claude's analysis appears in your local model's context.
```

**Key feature**: The `ask_*` tools have full tool access - the consulted model can read files, run commands, search memory, etc. You can chain consultations: ask Sonnet to ask GPT to analyze something.

The current provider is excluded (you don't ask yourself). Switch providers and the tools update automatically.

### Automatic Session Eviction

Shepherd supports automatic eviction for indefinite conversations with **any backend**:

- **Local backends**: Evicts when GPU KV cache fills
- **API backends**: Evicts when API returns context full error, then retries
- **Manual limit**: Use `--context-size N` to set a limit smaller than the backend's maximum

```bash
# Force eviction at 32K tokens even if backend supports more
./shepherd --provider azure --context-size 32768
```

**Eviction behavior**:
- Oldest messages first (LRU), protecting system prompt and current context
- Automatic archival to RAG database before eviction
- Seamless continuation - conversation keeps going

For local backend implementation details, see [docs/llamacpp.md](docs/llamacpp.md).

### Scheduling

Shepherd includes a cron-like scheduler that injects prompts into the session automatically. Works with CLI, TUI, and CLI server modes.

```bash
# Add a scheduled task (action-first for creating new)
shepherd sched add morning-news "0 9 * * *" "Get me the top 5 tech news headlines"

# List scheduled tasks
shepherd sched list

# View/modify schedules (name-first pattern)
shepherd sched morning-news show     # Show schedule details
shepherd sched morning-news disable  # Disable schedule
shepherd sched morning-news enable   # Enable schedule
shepherd sched morning-news remove   # Remove schedule
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
| `shepherd config <show\|set\|KEY\|KEY VALUE>` | View/modify configuration |
| `shepherd tools <list\|enable\|disable>` | Manage tools |
| `shepherd mcp <add\|remove\|list\|NAME show\|NAME test>` | Manage MCP servers |
| `shepherd smcp <add\|remove\|list>` | Manage SMCP servers (secure credentials) |
| `shepherd sched <list\|add\|remove\|enable\|disable\|show\|next>` | Scheduled tasks |
| `shepherd apikey <create\|list\|remove>` | API key management |
| `shepherd edit-system` | Edit system prompt in $EDITOR |

### Common Flags

| Flag | Description |
|------|-------------|
| `-p, --provider NAME` | Use specific provider |
| `-m, --model PATH` | Model name or file |
| `--backend TYPE` | Backend: llamacpp, tensorrt, openai, anthropic, gemini, grok, ollama, cli |
| `--context-size N` | Context window size (0 = auto) |
| `--max-tokens N` | Max generation tokens (-1 = max, 0 = auto, >0 = explicit) |
| `--prompt, -e TEXT` | Single query mode (run one query and exit) |
| `--server` | Start API server mode |
| `--cliserver` | Start CLI server mode |
| `--json` | JSON line-protocol frontend for machine integration |
| `--port N` | Server port (default: 8000) |
| `--use-tools` | Execute tools server-side in API server |
| `--server-tools` | Expose /v1/tools endpoints for tool discovery/execution |
| `--show-tool-calls BOOL` | Control streaming of tool call/result text to clients |
| `--notools` | Disable all tools |
| `--enable-tools PATTERN` | Enable tools matching glob pattern |
| `--disable-tools PATTERN` | Disable tools matching glob pattern |
| `--memtools` | Enable memory tools |
| `--nostream` | Disable streaming output |
| `--reasoning LEVEL` | Extended thinking: off, low, medium, high |
| `--stats` | Show performance stats (prefill/decode speed, KV cache) |
| `--tui` / `--no-tui` | Enable/disable TUI mode |
| `--nomcp` | Disable MCP server loading |
| `--norag` | Disable RAG entirely |
| `--nomemory` | Disable memory injection and extraction |
| `--nosched` | Disable scheduler |
| `--warmup` | Send warmup message before first prompt |
| `--flash-attn` | Enable flash attention (llama.cpp) |
| `--system-prompt TEXT` | Override system prompt |
| `--config msi --kv VAULT` | Load config from Azure Key Vault |
| `--config vault --kv ADDR` | Load config from HashiCorp Vault |

**Sampling overrides**: `--temperature`, `--top-p`, `--top-k`, `--freq` (frequency penalty), `--repeat-penalty`

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

Benchmarked with `scripts/openai_bench.py` (5 runs, 2048 max tokens, streaming, batch_size=1).

### Shepherd vs Other Servers (gpt-oss-120b Q4_K_M, 4x GPU pp=4, 32K context, f16 KV cache)

| Server | Tokens/sec | TTFT (ms) | ITL (ms) |
|--------|-----------|-----------|----------|
| **Shepherd** | **141.30** | 1063 | **7.62** |
| llama-server (standalone) | 124.52 | 952 | 8.03 |
| vLLM | 98.18 | 809 | 10.19 |

### Model Quantization Comparison (Shepherd, gpt-oss-120b, 4x GPU pp=4)

| Model / Quant | Tokens/sec | TTFT (ms) | ITL (ms) |
|---------------|-----------|-----------|----------|
| Q4_K_M | 141.30 | 1063 | 7.62 |
| Q4_K_M + flash-attn | 138.27 | 767 | 7.62 |
| MXFP4 | 128.72 | 794 | 8.21 |
| MXFP4 + flash-attn | 130.79 | 681 | 8.03 |

### Provider Configuration (Shepherd, gpt-oss-120b-heretic-v2-MXFP4, flash-attn, server-tools)

| Metric | Value |
|--------|-------|
| Tokens/sec (mean) | 130.85 |
| Tokens/sec (stddev) | 0.55 |
| TTFT (median) | 739 ms |
| ITL (mean) | 8.00 ms |
| Avg tokens generated | 2048 |

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
┌───────────────────────────────────────────────────────────────┐
│                          Frontend                              │
│  CLI, TUI, JSON, API Server, CLI Server                       │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────v────────────────────────────────────┐
│                     Session + Provider                         │
│  Message routing, provider switching, tool execution, RAG     │
└──────────────────────────┬────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────v────┐      ┌──────v─────┐     ┌─────v─────────┐
│  LlamaCpp  │      │  TensorRT  │     │ API Backends  │
│  Backend   │      │  Backend   │     │ (6 types)     │
└────────────┘      └────────────┘     └───────────────┘
```

For detailed architecture, see [docs/architecture.md](docs/architecture.md).

### Project Structure

```
shepherd/
├── main.cpp                  # Entry point, argument parsing, subcommand dispatch
├── frontend.cpp/h            # Frontend abstraction and event callback system
├── backend.cpp/h             # Backend base class
├── session.cpp/h             # Session management, eviction logic
├── session_manager.cpp/h     # Multi-session management
├── provider.cpp/h            # Provider configuration and switching
├── config.cpp/h              # Configuration (file, Azure KV, HashiCorp Vault)
├── rag.cpp/h                 # RAG interface and context injection
├── server.cpp/h              # HTTP server base class
├── auth.cpp/h                # API key authentication (JSON file, Azure MSI, PostgreSQL)
├── scheduler.cpp/h           # Cron-like task scheduler (SIGALRM-based)
├── memory_extraction.cpp/h   # Background fact extraction from conversations
├── http_client.cpp/h         # HTTP client with retry logic
├── generation_thread.cpp/h   # Threaded generation support
├── azure_msi.cpp/h           # Azure Managed Identity integration
├── hashicorp_vault.cpp/h     # HashiCorp Vault config loading
│
├── backends/
│   ├── api.cpp/h             # Base class for all API backends
│   ├── gpu.cpp/h             # Base class for local GPU backends
│   ├── llamacpp.cpp/h        # llama.cpp backend
│   ├── tensorrt.cpp/h        # TensorRT-LLM backend
│   ├── openai.cpp/h          # OpenAI / Azure OpenAI API
│   ├── anthropic.cpp/h       # Anthropic Claude API
│   ├── gemini.cpp/h          # Google Gemini API
│   ├── grok.cpp/h            # xAI Grok API
│   ├── ollama.cpp/h          # Ollama API
│   ├── cli_client.cpp/h      # CLI client (connects to remote CLI server)
│   ├── models.cpp/h          # Model family detection and context size database
│   ├── chat_template.cpp/h   # Chat template parsing (Jinja2)
│   ├── harmony.cpp/h         # GPT-OSS / Harmony format parser
│   └── factory.cpp/h         # Backend factory
│
├── frontends/
│   ├── cli.cpp/h             # Interactive CLI (replxx line editing)
│   ├── tui.cpp/h             # Full-screen TUI (ncurses)
│   ├── json_frontend.cpp/h   # JSON line-protocol (machine integration)
│   ├── api_server.cpp/h      # OpenAI-compatible HTTP API server
│   └── cli_server.cpp/h      # Persistent CLI session over HTTP + SSE
│
├── tools/                    # Tool implementations (core, filesystem, command,
│                             #   HTTP, JSON, memory, scheduler, MCP, remote, API)
├── mcp/                      # MCP/SMCP client, server, config, tool adapters
├── rag/                      # RAG database backends (SQLite, PostgreSQL)
├── scripts/                  # Benchmark and utility scripts
├── tests/                    # Unit and integration tests
├── docs/                     # Architecture and feature documentation
├── packaging/                # Debian packaging scripts
├── vendor/                   # Third-party: llama.cpp, replxx, tokenizers
├── CMakeLists.txt            # Build system
└── Makefile                  # Build wrapper
```

### Extending Shepherd

- **Adding backends**: See [docs/backends.md](docs/backends.md)
- **Adding tools**: See `tools/tool.h` for the tool interface
- **Architecture**: See [docs/architecture.md](docs/architecture.md) for the full system design
- **Frontend design**: See [docs/frontend.md](docs/frontend.md)

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
