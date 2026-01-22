#pragma once

#include "shepherd.h"
#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <map>
#include <functional>

/// @brief MCP server configuration entry
struct MCPServerEntry {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;

    nlohmann::json to_json() const;
    static MCPServerEntry from_json(const nlohmann::json& j);
};

/// @brief SMCP server configuration entry (MCP with secure credential injection)
struct SMCPServerEntry {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> credentials;  // Sent as JSON via SMCP handshake

    nlohmann::json to_json() const;
    static SMCPServerEntry from_json(const nlohmann::json& j);
};

/// @brief MCP configuration manager
class MCPConfig {
public:
    /// @brief Load MCP servers from config file
    static std::vector<MCPServerEntry> load(const std::string& config_path);

    /// @brief Save MCP servers to config file
    static bool save(const std::string& config_path, const std::vector<MCPServerEntry>& servers);

    /// @brief Add a new MCP server
    static bool add_server(const std::string& config_path, const MCPServerEntry& server);

    /// @brief Remove an MCP server by name
    static bool remove_server(const std::string& config_path, const std::string& name);

    /// @brief List all MCP servers
    /// @param check_health If true, test connection to each server
    static void list_servers(const std::string& config_path,
                             std::function<void(const std::string&)> callback,
                             bool check_health = false);

    /// @brief Check if server exists
    static bool server_exists(const std::string& config_path, const std::string& name);
};

// Common MCP command implementation (takes parsed args)
// Returns 0 on success, 1 on error
// callback: function to emit output
int handle_mcp_args(const std::vector<std::string>& args,
                    std::function<void(const std::string&)> callback);
