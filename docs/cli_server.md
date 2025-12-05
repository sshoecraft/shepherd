# CLI Server Module Documentation

## Overview

The CLI Server provides an HTTP server mode that executes tools locally while accepting prompts from remote clients. This enables a local machine with tools (filesystem access, shell commands) to serve as a backend for remote interfaces.

## Architecture

### Files

- `server/cli_server.h` - Class declaration
- `server/cli_server.cpp` - Implementation

### Class Structure

```cpp
class CLIServer : public Server {
public:
    CLIServer(const std::string& host, int port);
    ~CLIServer();

    int run(std::unique_ptr<Backend>& backend, Session& session) override;
};
```

## Usage

Start CLI server mode:

```bash
shepherd --cliserver --host 0.0.0.0 --port 8000
```

## Differences from API Server

| Feature | CLI Server | API Server |
|---------|------------|------------|
| Tool Execution | Local | Client-side |
| Purpose | Remote prompt input, local tools | Full OpenAI compatibility |
| Use Case | Secure tool environment | API compatibility |

## Security Considerations

The CLI server executes tools locally with the permissions of the shepherd process. Bind to localhost (`127.0.0.1`) unless network access is intended and secured.

## Version History

- **2.6.0** - Initial CLI server implementation
