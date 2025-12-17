
#include "shepherd.h"
#include "mcp/mcp.h"
#include "tools/tool.h"
#include "tools/tools.h"
#include "nlohmann/json.hpp"


MCP& MCP::instance() {
    static MCP manager;
    return manager;
}

bool MCP::initialize(Tools& tools) {
    dout(1) << "Initializing MCP Manager..." << std::endl;

    std::string mcp_json = config->mcp_config;
    dout(1) << "MCP config string: " + (mcp_json.empty() ? "(empty)" : mcp_json) << std::endl;
    if (mcp_json.empty()) {
        dout(1) << "No MCP servers configured" << std::endl;
        return true;
    }

    try {
        // Parse MCP server configurations
        nlohmann::json servers = nlohmann::json::parse(mcp_json);
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

        return initialize(tools, server_configs);

    } catch (const nlohmann::json::exception& e) {
        std::cerr << "Failed to parse MCP configuration: " + std::string(e.what()) << std::endl;
        return false;
    }
}

bool MCP::initialize(Tools& tools, const std::vector<ServerConfig>& server_configs) {
    dout(1) << "Initializing MCP Manager with " + std::to_string(server_configs.size()) + " servers" << std::endl;

    bool any_success = false;

    for (const auto& sconfig : server_configs) {
        dout(1) << "Connecting to MCP server: " + sconfig.name << std::endl;
        if (connect_server(tools, sconfig)) {
            any_success = true;
            dout(1) << "Successfully connected to MCP server: " + sconfig.name << std::endl;
        } else {
            dout(1) << std::string("WARNING: ") +"Failed to connect to MCP server: " + sconfig.name << std::endl;
        }
    }

    if (any_success) {
        dout(1) << "MCP Manager initialized with " +
                 std::to_string(clients_.size()) + " servers, " +
                 std::to_string(total_tools_) + " tools" << std::endl;
    } else {
        dout(1) << "No MCP servers could be initialized" << std::endl;
    }

    return any_success;
}

bool MCP::connect_server(Tools& tools, const ServerConfig& sconfig) {
    try {
        // Create server config
        MCPServer::Config server_config;
        server_config.name = sconfig.name;
        server_config.command = sconfig.command;
        server_config.args = sconfig.args;
        server_config.env = sconfig.env;

        // Create and start server
        auto server = std::make_unique<MCPServer>(server_config);
        server->start();  // Throws on error

        // Create client
        auto client = std::make_shared<MCPClient>(std::move(server));

        // Initialize protocol
        client->initialize();  // Throws on error

        // Discover tools
        auto mcp_tools = client->list_tools();
        dout(1) << "Discovered " + std::to_string(mcp_tools.size()) + " tools from " + server_config.name << std::endl;

        // Register tools with Tools instance
        for (const auto& mcp_tool : mcp_tools) {
            // Skip deprecated tools
            if (mcp_tool.description.find("DEPRECATED") != std::string::npos) {
                dout(1) << "Skipping deprecated MCP tool: " + mcp_tool.name << std::endl;
                continue;
            }

            // Create adapter and register as MCP tool
            auto adapter = std::make_unique<MCPToolAdapter>(client, mcp_tool);
            std::string tool_name = adapter->unsanitized_name();

            tools.register_tool(std::move(adapter), "mcp");
            dout(1) << "Registered MCP tool: " + tool_name + " (sanitized)" << std::endl;
            total_tools_++;
        }

        // Save client reference
        clients_.push_back(client);
        servers_by_name_[server_config.name] = client;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception connecting to MCP server " + sconfig.name + ": " + e.what() << std::endl;
        return false;
    }
}

void MCP::shutdown() {
    dout(1) << "Shutting down MCP Manager..." << std::endl;

    // Shutdown all clients
    for (auto& client : clients_) {
        try {
            // MCPClient destructor will handle shutdown
        } catch (const std::exception& e) {
            std::cerr << "Error shutting down MCP client: " + std::string(e.what()) << std::endl;
        }
    }

    clients_.clear();
    servers_by_name_.clear();
    total_tools_ = 0;

    dout(1) << "MCP Manager shutdown complete" << std::endl;
}

std::vector<std::string> MCP::get_server_names() const {
    std::vector<std::string> names;
    for (const auto& pair : servers_by_name_) {
        names.push_back(pair.first);
    }
    return names;
}

std::map<std::string, std::vector<MCPResource>> MCP::list_all_resources() const {
    std::map<std::string, std::vector<MCPResource>> all_resources;

    for (const auto& client : clients_) {
        try {
            std::string server_name = client->get_server_name();
            auto resources = client->list_resources();
            if (!resources.empty()) {
                all_resources[server_name] = resources;
                dout(1) << "Listed " + std::to_string(resources.size()) +
                         " resources from server: " + server_name << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to list resources from server: " +
                     client->get_server_name() + " - " + e.what() << std::endl;
        }
    }

    return all_resources;
}

std::vector<MCPResource> MCP::list_resources(const std::string& server_name) const {
    auto it = servers_by_name_.find(server_name);
    if (it == servers_by_name_.end()) {
        throw std::runtime_error("MCP server not found: " + server_name);
    }

    try {
        return it->second->list_resources();
    } catch (const std::exception& e) {
        std::cerr << "Failed to list resources from " + server_name + ": " + e.what() << std::endl;
        throw;
    }
}

nlohmann::json MCP::read_resource(const std::string& server_name, const std::string& uri) const {
    auto it = servers_by_name_.find(server_name);
    if (it == servers_by_name_.end()) {
        throw std::runtime_error("MCP server not found: " + server_name);
    }

    try {
        dout(1) << "Reading resource '" + uri + "' from server: " + server_name << std::endl;
        return it->second->read_resource(uri);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read resource from " + server_name + ": " + e.what() << std::endl;
        throw;
    }
}
