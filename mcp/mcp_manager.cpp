#include "mcp_manager.h"
#include "../tools/tool.h"
#include "../logger.h"
#include "../nlohmann/json.hpp"

using json = nlohmann::json;

MCPManager& MCPManager::instance() {
    static MCPManager manager;
    return manager;
}

bool MCPManager::initialize(const Config& config) {
    LOG_INFO("Initializing MCP Manager...");

    std::string mcp_json = config.get_mcp_config();
    if (mcp_json.empty()) {
        LOG_INFO("No MCP servers configured");
        return true;
    }

    try {
        // Parse MCP server configurations
        json servers = json::parse(mcp_json);
        std::vector<ServerConfig> server_configs;

        for (const auto& server : servers) {
            ServerConfig sc;
            sc.name = server.value("name", "unnamed");
            sc.command = server.value("command", "");

            if (server.contains("args") && server["args"].is_array()) {
                for (const auto& arg : server["args"]) {
                    sc.args.push_back(arg.get<std::string>());
                }
            }

            if (server.contains("env") && server["env"].is_object()) {
                for (auto it = server["env"].begin(); it != server["env"].end(); ++it) {
                    sc.env[it.key()] = it.value().get<std::string>();
                }
            }

            server_configs.push_back(sc);
        }

        return initialize(server_configs);

    } catch (const json::exception& e) {
        LOG_ERROR("Failed to parse MCP configuration: " + std::string(e.what()));
        return false;
    }
}

bool MCPManager::initialize(const std::vector<ServerConfig>& server_configs) {
    LOG_INFO("Initializing MCP Manager with " + std::to_string(server_configs.size()) + " servers");

    bool any_success = false;

    for (const auto& config : server_configs) {
        LOG_INFO("Connecting to MCP server: " + config.name);
        if (connect_server(config)) {
            any_success = true;
            LOG_INFO("Successfully connected to MCP server: " + config.name);
        } else {
            LOG_WARN("Failed to connect to MCP server: " + config.name);
        }
    }

    if (any_success) {
        LOG_INFO("MCP Manager initialized with " +
                 std::to_string(clients_.size()) + " servers, " +
                 std::to_string(total_tools_) + " tools");
    } else {
        LOG_DEBUG("No MCP servers could be initialized");
    }

    return any_success;
}

bool MCPManager::connect_server(const ServerConfig& config) {
    try {
        // Create server config
        MCPServer::Config server_config;
        server_config.name = config.name;
        server_config.command = config.command;
        server_config.args = config.args;
        server_config.env = config.env;

        // Create and start server
        auto server = std::make_unique<MCPServer>(server_config);
        server->start();  // Throws on error

        // Create client
        auto client = std::make_shared<MCPClient>(std::move(server));

        // Initialize protocol
        client->initialize();  // Throws on error

        // Discover tools
        auto mcp_tools = client->list_tools();
        LOG_DEBUG("Discovered " + std::to_string(mcp_tools.size()) + " tools from " + config.name);

        // Register tools with ToolRegistry
        auto& registry = ToolRegistry::instance();

        for (const auto& mcp_tool : mcp_tools) {
            // Skip deprecated tools
            if (mcp_tool.description.find("DEPRECATED") != std::string::npos) {
                LOG_DEBUG("Skipping deprecated MCP tool: " + mcp_tool.name);
                continue;
            }

            // Create adapter and register
            auto adapter = std::make_unique<MCPToolAdapter>(client, mcp_tool);
            std::string tool_name = adapter->unsanitized_name();

            registry.register_tool(std::move(adapter));
            LOG_DEBUG("Registered MCP tool: " + tool_name + " (sanitized)");
            total_tools_++;
        }

        // Save client reference
        clients_.push_back(client);
        servers_by_name_[config.name] = client;

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Exception connecting to MCP server " + config.name + ": " + e.what());
        return false;
    }
}

void MCPManager::shutdown() {
    LOG_INFO("Shutting down MCP Manager...");

    // Shutdown all clients
    for (auto& client : clients_) {
        try {
            // MCPClient destructor will handle shutdown
        } catch (const std::exception& e) {
            LOG_ERROR("Error shutting down MCP client: " + std::string(e.what()));
        }
    }

    clients_.clear();
    servers_by_name_.clear();
    total_tools_ = 0;

    LOG_INFO("MCP Manager shutdown complete");
}

std::vector<std::string> MCPManager::get_server_names() const {
    std::vector<std::string> names;
    for (const auto& pair : servers_by_name_) {
        names.push_back(pair.first);
    }
    return names;
}

std::map<std::string, std::vector<MCPResource>> MCPManager::list_all_resources() const {
    std::map<std::string, std::vector<MCPResource>> all_resources;

    for (const auto& client : clients_) {
        try {
            std::string server_name = client->get_server_name();
            auto resources = client->list_resources();
            if (!resources.empty()) {
                all_resources[server_name] = resources;
                LOG_DEBUG("Listed " + std::to_string(resources.size()) +
                         " resources from server: " + server_name);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to list resources from server: " +
                     client->get_server_name() + " - " + e.what());
        }
    }

    return all_resources;
}

std::vector<MCPResource> MCPManager::list_resources(const std::string& server_name) const {
    auto it = servers_by_name_.find(server_name);
    if (it == servers_by_name_.end()) {
        throw std::runtime_error("MCP server not found: " + server_name);
    }

    try {
        return it->second->list_resources();
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to list resources from " + server_name + ": " + e.what());
        throw;
    }
}

json MCPManager::read_resource(const std::string& server_name, const std::string& uri) const {
    auto it = servers_by_name_.find(server_name);
    if (it == servers_by_name_.end()) {
        throw std::runtime_error("MCP server not found: " + server_name);
    }

    try {
        LOG_DEBUG("Reading resource '" + uri + "' from server: " + server_name);
        return it->second->read_resource(uri);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to read resource from " + server_name + ": " + e.what());
        throw;
    }
}
