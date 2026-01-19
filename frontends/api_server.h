#pragma once

#include "server.h"
#include "backend.h"
#include "../session.h"
#include "../tools/tools.h"
#include "../backends/shared_oauth_cache.h"
#include <string>
#include <mutex>
#include <functional>
#include <memory>

/// @brief API Server - OpenAI-compatible HTTP API server
/// Stateless operation: Each request is independent, tools returned to client for execution
/// Also provides /v1/tools endpoints for tool discovery and execution via MCP proxy
class APIServer : public Server {
public:
    /// @brief Construct API server
    /// @param host Host to bind to
    /// @param port Port to listen on
    /// @param auth_mode Authentication mode: "none" or "json" (json requires valid API key)
    /// @param no_mcp If true, skip MCP tool initialization
    /// @param no_tools If true, skip all tool initialization
    APIServer(const std::string& host, int port,
              const std::string& auth_mode = "none",
              bool no_mcp = false,
              bool no_tools = false);
    ~APIServer();

    /// @brief Initialize tools and RAG
    void init(bool no_mcp = false, bool no_tools = false) override;

protected:
    /// @brief Register OpenAI-compatible API endpoints
    void register_endpoints() override;

private:
    // Mutex to serialize backend requests (single-threaded processing for GPU backends)
    std::mutex backend_mutex;

    // Shared OAuth cache for per-request API backends
    std::shared_ptr<SharedOAuthCache> shared_oauth_cache_;

    // Flag indicating if current provider is an API backend (supports per-request backends)
    bool is_api_provider_ = false;

    // Per-request output routing - set before generate, cleared after
    // The callback routes events through this function when set
    std::function<bool(CallbackEvent, const std::string&, const std::string&, const std::string&)> request_handler;

    /// @brief Handle chat completion request (standard OpenAI behavior)
    /// @param req HTTP request
    /// @param res HTTP response
    /// @param request Parsed JSON request body
    void handle_chat_request(const httplib::Request& req,
                             httplib::Response& res,
                             const nlohmann::json& request);

    /// @brief Extract Bearer token from Authorization header
    /// @param req HTTP request
    /// @return API key or empty string if not present
    std::string extract_bearer_token(const httplib::Request& req) const;
};
