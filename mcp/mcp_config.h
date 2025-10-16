#pragma once

#include "../nlohmann/json.hpp"
#include <string>
#include <vector>
#include <map>

using json = nlohmann::json;

/// @brief MCP server configuration entry
struct MCPServerEntry {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;

    json to_json() const;
    static MCPServerEntry from_json(const json& j);
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
    static void list_servers(const std::string& config_path, bool check_health = false);

    /// @brief Check if server exists
    static bool server_exists(const std::string& config_path, const std::string& name);
};
