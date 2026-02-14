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
#include <thread>
#include <algorithm>
#include <set>

// Forward declaration
class Tools;

class MCP {
public:
    struct ServerConfig {
        std::string name;
        std::string command;
        std::vector<std::string> args;
        std::map<std::string, std::string> env;
        std::map<std::string, std::string> smcp_credentials;
    };

    static MCP& instance();

    bool initialize(Tools& tools);
    bool initialize(Tools& tools, const std::vector<ServerConfig>& server_configs);
    void shutdown();

    // Collect all active pipe FDs from running MCP/SMCP servers
    std::set<int> get_active_fds() const;

    std::vector<std::shared_ptr<MCPClient>> clients;
    std::map<std::string, std::shared_ptr<MCPClient>> servers_by_name;
    size_t total_tools = 0;

    struct ServerInitResult {
        std::shared_ptr<MCPClient> client;
        std::string server_name;
        std::vector<MCPTool> tools;
        bool success = false;
    };

    ServerInitResult init_server(const ServerConfig& config);
    void register_server(Tools& tools, ServerInitResult& result);

    std::map<std::string, std::vector<MCPResource>> list_all_resources() const;
    std::vector<MCPResource> list_resources(const std::string& server_name) const;
    nlohmann::json read_resource(const std::string& server_name, const std::string& uri) const;

private:
    MCP() = default;
    ~MCP() { shutdown(); }

    MCP(const MCP&) = delete;
    MCP& operator=(const MCP&) = delete;
};
