#pragma once

#include "server.h"
#include "../backends/backend.h"
#include "../session.h"
#include "../tools/tools.h"
#include <string>
#include <memory>

/// @brief CLI Server - HTTP server that executes tools locally
class CLIServer : public Server {
public:
    CLIServer(const std::string& host, int port);
    ~CLIServer();

    /// @brief Initialize tools and RAG
    void init(Session& session,
              bool no_mcp = false,
              bool no_tools = false,
              const std::string& provider_name = "") override;

    /// @brief Run the CLI server
    int run(Session& session) override;

    /// Tool management
    Tools tools;
};

/// @brief Run CLI server mode - HTTP server that executes tools locally
/// @param backend The backend to use for generation
/// @param session The session with tools and system prompt configured
/// @param host Host address to bind to (e.g. "0.0.0.0")
/// @param port Port number to listen on
/// @param tools Tools instance for tool execution
/// @return 0 on success, non-zero on error
int run_cli_server(std::unique_ptr<Backend>& backend, Session& session,
                   const std::string& host, int port, Tools& tools);
