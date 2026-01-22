#pragma once

#include "shepherd.h"

#include "mcp/mcp_config.h"
#include "mcp/mcp_client.h"
#include "mcp/mcp_tool.h"
#include "tools/tool.h"
#include "config.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <memory>
#include <string>

// Forward declaration
class Tools;

/// @brief Manages multiple MCP servers and their tools
/// Loads MCP servers from config and registers their tools with Tools instance
class MCP {
public:
    /// @brief Server configuration structure
    struct ServerConfig {
        std::string name;
        std::string command;
        std::vector<std::string> args;
        std::map<std::string, std::string> env;
        std::map<std::string, std::string> smcp_credentials;  // For SMCP servers
    };

    static MCP& instance();

    /// @brief Initialize MCP servers from config
    /// @param tools Tools instance to register MCP tools with
    /// @return True if at least one server initialized successfully
    bool initialize(Tools& tools);

    /// @brief Initialize from explicit server configs (for testing)
    /// @param tools Tools instance to register MCP tools with
    /// @param server_configs List of server configurations
    /// @return True if at least one server initialized successfully
    bool initialize(Tools& tools, const std::vector<ServerConfig>& server_configs);

    /// @brief Shutdown all MCP servers
    void shutdown();

    /// @brief Get number of active MCP servers
    size_t get_server_count() const { return clients_.size(); }

    /// @brief Get total number of MCP tools registered
    size_t get_tool_count() const { return total_tools_; }

    /// @brief Get list of server names
    std::vector<std::string> get_server_names() const;

    /// @brief List all resources from all connected MCP servers
    /// @return Map of server name to list of resources
    std::map<std::string, std::vector<MCPResource>> list_all_resources() const;

    /// @brief List resources from a specific MCP server
    /// @param server_name Name of the server
    /// @return List of resources from that server
    std::vector<MCPResource> list_resources(const std::string& server_name) const;

    /// @brief Read a resource from a specific MCP server
    /// @param server_name Name of the server
    /// @param uri Resource URI
    /// @return Resource content as JSON
    nlohmann::json read_resource(const std::string& server_name, const std::string& uri) const;

private:
    MCP() = default;
    ~MCP() { shutdown(); }

    // Disable copy
    MCP(const MCP&) = delete;
    MCP& operator=(const MCP&) = delete;

    /// @brief Connect to a single MCP server
    /// @param tools Tools instance to register MCP tools with
    /// @param config Server configuration
    /// @return True if successful
    bool connect_server(Tools& tools, const ServerConfig& config);

    std::vector<std::shared_ptr<MCPClient>> clients_;
    std::map<std::string, std::shared_ptr<MCPClient>> servers_by_name_;
    size_t total_tools_ = 0;
};
