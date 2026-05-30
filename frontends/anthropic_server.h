#pragma once

#include "server.h"
#include "backend.h"
#include "../session.h"
#include "../backends/shared_oauth_cache.h"
#include <string>
#include <mutex>
#include <functional>
#include <memory>

/// @brief Anthropic API Server - Anthropic Messages API compatible HTTP server
/// Implements /v1/messages endpoint for Claude Code and other Anthropic API clients
class AnthropicServer : public Server {
public:
    /// @brief Construct Anthropic API server
    /// @param host Host to bind to
    /// @param port Port to listen on
    /// @param ssl_cert Path to TLS certificate (empty for plain HTTP)
    /// @param ssl_key Path to TLS private key (empty for plain HTTP)
    /// @param passthrough If true, proxy requests directly without conversion
    AnthropicServer(const std::string& host, int port,
                    const std::string& ssl_cert = "",
                    const std::string& ssl_key = "",
                    bool passthrough = false);
    ~AnthropicServer();

    /// @brief Initialize tools and RAG
    void init(const FrontendFlags& flags) override;

protected:
    /// @brief Register Anthropic-compatible API endpoints
    void register_endpoints() override;

private:
    // Passthrough mode - proxy requests directly to upstream without conversion
    bool passthrough_mode = false;

    // Shared OAuth cache for per-request API backends
    std::shared_ptr<SharedOAuthCache> shared_oauth_cache;

    // Flag indicating if current provider is an API backend
    bool is_api_provider = false;

    // Per-request output routing
    std::function<bool(CallbackEvent, const std::string&, const std::string&, const std::string&)> request_handler;

    /// @brief Extract Bearer/x-api-key token from request
    std::string extract_api_key(const httplib::Request& req) const;

    /// @brief Convert OpenAI-style messages to Anthropic format
    nlohmann::json convert_to_anthropic_response(const std::string& content,
                                                  const std::string& model,
                                                  const std::string& stop_reason,
                                                  int input_tokens,
                                                  int output_tokens);
};
