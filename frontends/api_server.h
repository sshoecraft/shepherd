#pragma once

#include "server.h"
#include "backend.h"
#include "../session.h"
#include "../session_manager.h"
#include <string>
#include <mutex>
#include <functional>
#include <memory>

/// @brief API Server - OpenAI-compatible HTTP API server
/// Supports both stateless (standard OpenAI) and stateful (server-side tools) modes
/// - Stateless: Each request is independent, tools returned to client for execution
/// - Stateful: Session persists per API key, tools executed server-side
class APIServer : public Server {
public:
    /// @brief Construct API server
    /// @param host Host to bind to
    /// @param port Port to listen on
    /// @param auth_mode Authentication mode: "none" or "json" (json requires valid API key)
    /// @param no_mcp If true, skip MCP tool initialization for sessions
    /// @param no_tools If true, skip all tool initialization for sessions
    APIServer(const std::string& host, int port,
              const std::string& auth_mode = "none",
              bool no_mcp = false,
              bool no_tools = false);
    ~APIServer();

protected:
    /// @brief Register OpenAI-compatible API endpoints
    void register_endpoints() override;

    /// @brief Initialize session manager after backend is connected
    void on_server_start() override;

    /// @brief Cleanup session manager
    void on_server_stop() override;

    /// @brief Add session info to status response
    void add_status_info(nlohmann::json& status) override;

private:
    // Mutex to serialize backend requests (single-threaded processing)
    std::mutex backend_mutex;

    // Per-request output routing - set before generate, cleared after
    // The callback routes events through this function when set
    std::function<bool(CallbackEvent, const std::string&, const std::string&, const std::string&)> request_handler;

    // Multi-tenant session support
    std::unique_ptr<SessionManager> session_manager;
    bool no_mcp;
    bool no_tools;

    /// @brief Handle stateful request (server-side tools)
    /// @param req HTTP request
    /// @param res HTTP response
    /// @param managed Session for this API key
    /// @param request Parsed JSON request body
    void handle_stateful_request(const httplib::Request& req,
                                  httplib::Response& res,
                                  ManagedSession* managed,
                                  const nlohmann::json& request);

    /// @brief Handle stateless request (standard OpenAI behavior)
    /// @param req HTTP request
    /// @param res HTTP response
    /// @param request Parsed JSON request body
    void handle_stateless_request(const httplib::Request& req,
                                   httplib::Response& res,
                                   const nlohmann::json& request);

    /// @brief Extract Bearer token from Authorization header
    /// @param req HTTP request
    /// @return API key or empty string if not present
    std::string extract_bearer_token(const httplib::Request& req) const;
};
