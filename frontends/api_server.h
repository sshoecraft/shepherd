#pragma once

#include "server.h"
#include "backend.h"
#include "../session.h"
#include <string>
#include <mutex>
#include <functional>

/// @brief API Server - OpenAI-compatible HTTP API server
class APIServer : public Server {
public:
    APIServer(const std::string& host, int port, const std::string& auth_mode = "none");
    ~APIServer();

protected:
    /// @brief Register OpenAI-compatible API endpoints
    void register_endpoints() override;

private:
    // Mutex to serialize backend requests (single-threaded processing)
    std::mutex backend_mutex;

    // Per-request output routing - set before generate, cleared after
    // The callback routes events through this function when set
    std::function<bool(CallbackEvent, const std::string&, const std::string&, const std::string&)> request_handler;
};
