#pragma once

#include "server.h"
#include "../backends/backend.h"
#include "../session.h"
#include "../llama.cpp/vendor/cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"
#include <string>
#include <atomic>
#include <random>

/// @brief API Server - OpenAI-compatible HTTP API server
class APIServer : public Server {
public:
    APIServer(const std::string& host, int port);
    ~APIServer();

    /// @brief Run the API server
    int run(Session& session) override;
};

/// @brief Run API server
/// @param backend The backend to use for generation
/// @param host Host address to bind to (e.g. "0.0.0.0")
/// @param port Port number to listen on
/// @return 0 on success, non-zero on error
int run_api_server(Backend* backend, const std::string& host, int port);
