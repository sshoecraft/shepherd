#pragma once

#include "server.h"
#include "backend.h"
#include "../session.h"
#include <string>
#include <mutex>

/// @brief API Server - OpenAI-compatible HTTP API server
class APIServer : public Server {
public:
    APIServer(const std::string& host, int port);
    ~APIServer();

protected:
    /// @brief Register OpenAI-compatible API endpoints
    void register_endpoints() override;

private:
    // Mutex to serialize backend requests (single-threaded processing)
    std::mutex backend_mutex;
};
