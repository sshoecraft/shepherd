#pragma once

#include "backends/backend.h"
#include <memory>
#include <string>

/// @brief Run Shepherd in HTTP server mode with OpenAI-compatible API
/// @param backend The initialized backend instance to use for inference
/// @param server_host Host address to bind the server to
/// @param server_port Port number to listen on
/// @return 0 on success, non-zero on error
int run_server(std::unique_ptr<Backend>& backend, const std::string& server_host, int server_port);
