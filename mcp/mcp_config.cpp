#include "shepherd.h"
#include "mcp_config.h"
#include "mcp_server.h"
#include "mcp_client.h"
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

nlohmann::json SMCPServerEntry::to_json() const {
    nlohmann::json j = {
        {"name", name},
        {"command", command}
    };

    if (!args.empty()) {
        j["args"] = args;
    }

    if (!credentials.empty()) {
        j["credentials"] = credentials;
    }

    return j;
}

SMCPServerEntry SMCPServerEntry::from_json(const nlohmann::json& j) {
    SMCPServerEntry entry;
    entry.name = j.value("name", "");
    entry.command = j.value("command", "");

    if (j.contains("args") && j["args"].is_array()) {
        for (const auto& arg : j["args"]) {
            entry.args.push_back(arg.get<std::string>());
        }
    }

    if (j.contains("credentials") && j["credentials"].is_object()) {
        for (auto it = j["credentials"].begin(); it != j["credentials"].end(); ++it) {
            entry.credentials[it.key()] = it.value().get<std::string>();
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
        std::cerr << "Failed to parse MCP config: " + std::string(e.what()) << std::endl;
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
            std::cerr << "Failed to parse existing config: " + std::string(e.what()) << std::endl;
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
        std::cerr << "Failed to open config file for writing: " + config_path << std::endl;
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

void MCPConfig::list_servers(const std::string& config_path,
                             std::function<void(const std::string&)> callback,
                             bool check_health) {
    auto servers = load(config_path);

    if (servers.empty()) {
        callback("No MCP servers configured\n");
        return;
    }

    if (check_health) {
        callback("Checking MCP server health...\n\n");
    }

    for (const auto& server : servers) {
        std::string line = server.name + ": " + server.command;
        for (const auto& arg : server.args) {
            line += " " + arg;
        }

        if (check_health) {
            // Test server connection
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

                line += " - ✓ Connected";
            } catch (const std::exception& e) {
                line += " - ✗ Failed";
            }
        }

        callback(line + "\n");
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

// Common MCP command implementation
int handle_mcp_args(const std::vector<std::string>& args,
                    std::function<void(const std::string&)> callback) {
    std::string config_path = Config::get_default_config_path();

    // No args shows servers
    if (args.empty()) {
        MCPConfig::list_servers(config_path, callback, true);  // Check health
        return 0;
    }

    // Determine if first arg is an action (action-first) or a name (name-first)
    std::string first_arg = args[0];
    bool action_first = (first_arg == "list" || first_arg == "add" ||
                         first_arg == "help" || first_arg == "--help" || first_arg == "-h");

    std::string name;
    std::string subcmd;

    if (action_first) {
        subcmd = first_arg;
        name = (args.size() >= 2) ? args[1] : "";
    } else {
        // Name-first: mcp NAME [action] [args...]
        name = first_arg;
        subcmd = (args.size() >= 2) ? args[1] : "help";  // No action = show help
    }

    // Check for read-only mode (Key Vault config) for modifying commands
    if (config && config->is_read_only()) {
        if (subcmd == "add" || subcmd == "remove") {
            callback("Error: Cannot modify MCP servers in read-only mode (config from Key Vault)\n");
            return 1;
        }
    }

    if (subcmd == "help" || subcmd == "--help" || subcmd == "-h") {
        if (!name.empty()) {
            callback("Usage: shepherd mcp " + name + " <action>\n"
                "\nActions:\n"
                "  show    - Show server details\n"
                "  test    - Test server connection\n"
                "  remove  - Remove server\n");
        } else {
            callback("Usage: shepherd mcp <name> <action> [args...]\n"
                "\nActions (after name):\n"
                "  show    - Show server details\n"
                "  test    - Test server connection\n"
                "  remove  - Remove server\n"
                "\nOther commands:\n"
                "  list    - List all servers\n"
                "  add <name> <command> [args...] [-e KEY=VALUE ...]\n");
        }
        return 0;
    }

    if (subcmd == "list") {
        MCPConfig::list_servers(config_path, callback, true);
        return 0;
    }

    if (subcmd == "add") {
        // Action-first: mcp add NAME COMMAND [args...]
        if (args.size() < 3) {
            callback("Usage: shepherd mcp add <name> <command> [args...] [-e KEY=VALUE ...]\n");
            return 1;
        }

        MCPServerEntry server;
        server.name = args[1];
        server.command = args[2];

        // Parse arguments and environment variables
        for (size_t i = 3; i < args.size(); i++) {
            const std::string& arg = args[i];

            if (arg == "-e" || arg == "--env") {
                // Next arg should be KEY=VALUE
                if (i + 1 < args.size()) {
                    i++;
                    std::string env_pair = args[i];
                    size_t eq_pos = env_pair.find('=');
                    if (eq_pos != std::string::npos) {
                        std::string key = env_pair.substr(0, eq_pos);
                        std::string value = env_pair.substr(eq_pos + 1);
                        server.env[key] = value;
                    } else {
                        callback("Warning: Invalid env format (use KEY=VALUE): " + env_pair + "\n");
                    }
                }
            } else {
                server.args.push_back(arg);
            }
        }

        if (MCPConfig::add_server(config_path, server)) {
            callback("Added MCP server '" + server.name + "'\n");
            return 0;
        }
        return 1;
    }

    if (subcmd == "show") {
        if (name.empty()) {
            callback("Usage: shepherd mcp <name> show\n");
            return 1;
        }

        auto servers = MCPConfig::load(config_path);
        for (const auto& server : servers) {
            if (server.name == name) {
                callback("=== MCP Server: " + name + " ===\n");
                callback("command = " + server.command + "\n");
                callback("args = [");
                for (size_t i = 0; i < server.args.size(); i++) {
                    if (i > 0) callback(", ");
                    callback("\"" + server.args[i] + "\"");
                }
                callback("]\n");
                if (!server.env.empty()) {
                    callback("env:\n");
                    for (const auto& [k, v] : server.env) {
                        callback("  " + k + " = " + v + "\n");
                    }
                }
                return 0;
            }
        }
        callback("MCP server '" + name + "' not found\n");
        return 1;
    }

    if (subcmd == "remove") {
        if (name.empty()) {
            callback("Usage: shepherd mcp <name> remove\n");
            return 1;
        }

        if (MCPConfig::remove_server(config_path, name)) {
            callback("Removed MCP server '" + name + "'\n");
            return 0;
        }
        return 1;
    }

    if (subcmd == "test") {
        if (name.empty()) {
            callback("Usage: shepherd mcp <name> test\n");
            return 1;
        }

        auto servers = MCPConfig::load(config_path);

        for (const auto& server : servers) {
            if (server.name == name) {
                callback("Testing MCP server '" + name + "'...\n");
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

                    auto tools = client.list_tools();
                    callback("Connected! Server provides " + std::to_string(tools.size()) + " tools:\n");
                    for (const auto& tool : tools) {
                        callback("  - " + tool.name + ": " + tool.description + "\n");
                    }
                    return 0;
                } catch (const std::exception& e) {
                    callback("Failed to connect: " + std::string(e.what()) + "\n");
                    return 1;
                }
            }
        }

        callback("MCP server '" + name + "' not found\n");
        return 1;
    }

    callback("Unknown mcp command: " + subcmd + "\n");
    callback("Use 'shepherd mcp help' to see available commands\n");
    return 1;
}
