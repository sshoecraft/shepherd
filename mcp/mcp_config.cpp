#include "mcp_config.h"
#include "mcp_server.h"
#include "mcp_client.h"
#include "../logger.h"
#include <fstream>
#include <iostream>
#include <iomanip>

nlohmann::json MCPServerEntry::to_json() const {
    nlohmann::json j = {
        {"name", name},
        {"command", command},
        {"args", args}
    };

    if (!env.empty()) {
        j["env"] = env;
    }

    return j;
}

MCPServerEntry MCPServerEntry::from_json(const nlohmann::json& j) {
    MCPServerEntry entry;
    entry.name = j.value("name", "");
    entry.command = j.value("command", "");

    if (j.contains("args") && j["args"].is_array()) {
        for (const auto& arg : j["args"]) {
            entry.args.push_back(arg.get<std::string>());
        }
    }

    if (j.contains("env") && j["env"].is_object()) {
        for (auto it = j["env"].begin(); it != j["env"].end(); ++it) {
            entry.env[it.key()] = it.value().get<std::string>();
        }
    }

    return entry;
}

std::vector<MCPServerEntry> MCPConfig::load(const std::string& config_path) {
    std::vector<MCPServerEntry> servers;

    std::ifstream file(config_path);
    if (!file.is_open()) {
        return servers;
    }

    try {
        nlohmann::json config = nlohmann::json::parse(file);

        if (config.contains("mcp_servers") && config["mcp_servers"].is_array()) {
            for (const auto& server_json : config["mcp_servers"]) {
                servers.push_back(MCPServerEntry::from_json(server_json));
            }
        }
    } catch (const nlohmann::json::exception& e) {
        LOG_ERROR("Failed to parse MCP config: " + std::string(e.what()));
    }

    return servers;
}

bool MCPConfig::save(const std::string& config_path, const std::vector<MCPServerEntry>& servers) {
    std::ifstream infile(config_path);
    nlohmann::json config;

    if (infile.is_open()) {
        try {
            config = nlohmann::json::parse(infile);
        } catch (const nlohmann::json::exception& e) {
            LOG_ERROR("Failed to parse existing config: " + std::string(e.what()));
            config = nlohmann::json::object();
        }
        infile.close();
    } else {
        config = nlohmann::json::object();
    }

    // Update mcp_servers array
    nlohmann::json mcp_array = nlohmann::json::array();
    for (const auto& server : servers) {
        mcp_array.push_back(server.to_json());
    }
    config["mcp_servers"] = mcp_array;

    // Write back to file
    std::ofstream outfile(config_path);
    if (!outfile.is_open()) {
        LOG_ERROR("Failed to open config file for writing: " + config_path);
        return false;
    }

    outfile << config.dump(2) << std::endl;
    return true;
}

bool MCPConfig::add_server(const std::string& config_path, const MCPServerEntry& server) {
    auto servers = load(config_path);

    // Check for duplicates
    for (const auto& existing : servers) {
        if (existing.name == server.name) {
            std::cerr << "Error: MCP server '" << server.name << "' already exists" << std::endl;
            return false;
        }
    }

    servers.push_back(server);
    return save(config_path, servers);
}

bool MCPConfig::remove_server(const std::string& config_path, const std::string& name) {
    auto servers = load(config_path);
    bool found = false;

    servers.erase(
        std::remove_if(servers.begin(), servers.end(),
            [&name, &found](const MCPServerEntry& s) {
                if (s.name == name) {
                    found = true;
                    return true;
                }
                return false;
            }),
        servers.end()
    );

    if (!found) {
        std::cerr << "Error: MCP server '" << name << "' not found" << std::endl;
        return false;
    }

    return save(config_path, servers);
}

void MCPConfig::list_servers(const std::string& config_path, bool check_health) {
    auto servers = load(config_path);

    if (servers.empty()) {
        std::cout << "No MCP servers configured" << std::endl;
        return;
    }

    if (check_health) {
        std::cout << "Checking MCP server health...\n" << std::endl;
    }

    for (const auto& server : servers) {
        std::cout << server.name << ": " << server.command;
        for (const auto& arg : server.args) {
            std::cout << " " << arg;
        }

        if (check_health) {
            // Test server connection
            std::string status;
            try {
                MCPServer::Config srv_config;
                srv_config.name = server.name;
                srv_config.command = server.command;
                srv_config.args = server.args;
                srv_config.env = server.env;

                auto srv = std::make_unique<MCPServer>(srv_config);
                srv->start();

                MCPClient client(std::move(srv));
                client.initialize();

                status = " - ✓ Connected";
            } catch (const std::exception& e) {
                status = " - ✗ Failed";
            }
            std::cout << status;
        }

        std::cout << std::endl;
    }
}

bool MCPConfig::server_exists(const std::string& config_path, const std::string& name) {
    auto servers = load(config_path);
    for (const auto& server : servers) {
        if (server.name == name) {
            return true;
        }
    }
    return false;
}
