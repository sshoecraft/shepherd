#pragma once

#include "tool.h"
#include "../http_client.h"
#include "nlohmann/json.hpp"
#include <string>
#include <memory>

class Tools;

/// @brief Proxy for a tool that executes on a remote server via /v1/tools/execute
/// Created by fetching tool definitions from a remote /v1/tools endpoint
class RemoteToolProxy : public Tool {
public:
    /// @brief Construct a proxy for a remote tool
    /// @param name Tool name
    /// @param description Tool description
    /// @param parameters_schema JSON schema for parameters
    /// @param server_url Base URL of tools server (e.g., "http://localhost:8000/v1")
    /// @param api_key API key for authentication (Bearer token)
    RemoteToolProxy(const std::string& name,
                    const std::string& description,
                    const nlohmann::json& parameters_schema,
                    const std::string& server_url,
                    const std::string& api_key);

    ~RemoteToolProxy() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;

private:
    std::string tool_name;
    std::string tool_description;
    nlohmann::json params_schema;
    std::string server_url;
    std::string api_key;

    // Lazy-initialized HTTP client
    mutable std::unique_ptr<HttpClient> http_client;
    void ensure_client() const;
};

/// @brief Fetch remote tools from server and register them as proxies
/// @param tools The Tools instance to register proxies into
/// @param server_url Base URL of tools server (e.g., "http://localhost:8000/v1")
/// @param api_key API key for authentication
/// @return Number of tools registered, or -1 on error
int register_remote_tools(Tools& tools, const std::string& server_url, const std::string& api_key);
