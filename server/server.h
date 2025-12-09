#pragma once

#include "../frontend.h"
#include "../llama.cpp/vendor/cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"
#include <memory>
#include <string>
#include <atomic>
#include <thread>
#include <chrono>

/// @brief Base class for all server frontends (API Server, CLI Server)
/// Manages HTTP server lifecycle, control socket, and common endpoints
class Server : public Frontend {
public:
    Server(const std::string& host, int port, const std::string& server_type);
    virtual ~Server();

    /// @brief Run the server - starts TCP and control socket listeners
    /// @param session Session for the server
    /// @return 0 on success, non-zero on error
    int run(Session& session) override;

    /// @brief Initiate graceful shutdown
    void shutdown();

protected:
    /// @brief Register server-specific endpoints on the TCP server
    /// Subclasses must implement this to add their endpoints
    /// @param session Session reference for endpoint handlers
    virtual void register_endpoints(Session& session) = 0;

    /// @brief Add subclass-specific info to status response
    /// @param status JSON object to add fields to
    virtual void add_status_info(nlohmann::json& status) {}

    /// @brief Called before server starts listening (after endpoints registered)
    /// Subclasses can override to start background threads, etc.
    virtual void on_server_start() {}

    /// @brief Called after server stops listening (before cleanup)
    /// Subclasses can override to stop background threads, etc.
    virtual void on_server_stop() {}

    // TCP server for main API endpoints
    httplib::Server tcp_server;

    // Control socket server (Unix domain socket)
    httplib::Server control_server;

    // Server state
    std::atomic<bool> running{true};
    std::chrono::steady_clock::time_point start_time;
    std::atomic<uint64_t> requests_processed{0};

    // Configuration
    std::string host;
    int port;
    std::string server_type;
    std::string control_socket_path;

private:
    /// @brief Register common endpoints (/health, /status)
    void register_common_endpoints();

    /// @brief Register control socket endpoints (/shutdown, /status)
    void register_control_endpoints();

    /// @brief Setup and start the control socket listener
    /// @return true if successful, false on error
    bool start_control_socket();

    /// @brief Cleanup control socket on shutdown
    void cleanup_control_socket();

    // Control socket thread
    std::thread control_thread;
};

