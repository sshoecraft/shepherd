#pragma once

#include "server.h"
#include "../tools/tools.h"
#include <string>

/// @brief OpenAPI tool server - exposes the aggregated MCP/SMCP tool registry as
/// an OpenAPI 3.1 service. Does NOT load any LLM; never reaches a provider.
/// Designed for Open WebUI's "tool server" integration.
class OpenAPIServer : public Server {
public:
    OpenAPIServer(const std::string& host, int port,
                  const std::string& ssl_cert = "",
                  const std::string& ssl_key = "",
                  const std::string& base_url = "");
    ~OpenAPIServer();

    /// @brief Initialize tools (always local — we're the tool server).
    void init(const FrontendFlags& flags) override;

    /// @brief Override Server::run() to skip provider/backend connection.
    int run(Provider* cmdline_provider = nullptr) override;

protected:
    void register_endpoints() override;

private:
    // Optional explicit URL for the OpenAPI spec's servers[0].url
    std::string base_url;

    // Build the full OpenAPI 3.1 document. Recomputed per request — cheap.
    nlohmann::json build_openapi_document(const httplib::Request& req);

    // Single per-tool POST handler, shared by every registered route.
    void handle_tool_call(const std::string& tool_name,
                          const httplib::Request& req,
                          httplib::Response& res);

    // Extract Bearer token from Authorization header.
    std::string extract_bearer_token(const httplib::Request& req) const;
};
