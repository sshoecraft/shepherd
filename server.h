#pragma once

#include "frontend.h"
#include "auth.h"
#include "llama.cpp/vendor/cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"
#include <memory>
#include <string>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>

/// @brief Control client for communicating with shepherd server via Unix socket
/// Used by `shepherd ctl` command to query/control a running server
class ControlClient {
public:
    ControlClient(const std::string& socket_path);

    /// @brief Get server status
    /// @return JSON response or error object with "error" field
    nlohmann::json get_status();

    /// @brief Request server shutdown
    /// @return JSON response or error object with "error" field
    nlohmann::json shutdown();

    /// @brief Check if socket exists and is accessible
    bool socket_exists() const;

private:
    /// @brief Send HTTP request over Unix socket
    std::string http_request(const std::string& method, const std::string& path);

    std::string socket_path;
};

/// @brief Handle ctl subcommand (shepherd ctl status, shepherd ctl shutdown)
/// @param args Arguments after "ctl"
/// @return Exit code
int handle_ctl_args(const std::vector<std::string>& args);

/// @brief Base class for all server frontends (API Server, CLI Server)
/// Manages HTTP server lifecycle, control socket, and common endpoints
class Server : public Frontend {
public:
    Server(const std::string& host, int port, const std::string& server_type,
           const std::string& auth_mode = "none");
    virtual ~Server();

    /// @brief Run the server - starts TCP and control socket listeners
    /// @param cmdline_provider Optional provider from command-line override
    /// @return 0 on success, non-zero on error
    int run(Provider* cmdline_provider = nullptr) override;

    /// @brief Initiate graceful shutdown
    void shutdown();

protected:
    /// @brief Register server-specific endpoints on the TCP server
    /// Subclasses must implement this to add their endpoints
    /// Uses frontend's session member
    virtual void register_endpoints() = 0;

    /// @brief Add subclass-specific info to status response
    /// @param status JSON object to add fields to
    virtual void add_status_info(nlohmann::json& status) {}

    /// @brief Called before server starts listening (after endpoints registered)
    /// Subclasses can override to start background threads, etc.
    virtual void on_server_start() {}

    /// @brief Called after server stops listening (before cleanup)
    /// Subclasses can override to stop background threads, etc.
    virtual void on_server_stop() {}

    /// @brief Called when shutdown is requested (before tcp_server.stop())
    /// Subclasses can override to signal threads to exit
    virtual void on_shutdown() {}

    /// @brief Check if request is authenticated (when auth is enabled)
    /// @param req HTTP request
    /// @param res HTTP response (set to 401 on failure)
    /// @return true if authorized, false otherwise (response already set)
    bool check_auth(const httplib::Request& req, httplib::Response& res);

    // API key authentication store
    std::unique_ptr<KeyStore> key_store;

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

