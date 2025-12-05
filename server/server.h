#pragma once

#include "../frontend.h"
#include <memory>
#include <string>

/// @brief Base class for all server frontends (API Server, CLI Server)
class Server : public Frontend {
public:
    Server(const std::string& host, int port);
    virtual ~Server();

protected:
    std::string host;
    int port;
};

/// @brief Run Shepherd in HTTP server mode with OpenAI-compatible API
/// @param backend The initialized backend instance to use for inference
/// @param server_host Host address to bind the server to
/// @param server_port Port number to listen on
/// @return 0 on success, non-zero on error
int run_server(std::unique_ptr<Backend>& backend, const std::string& server_host, int server_port);
