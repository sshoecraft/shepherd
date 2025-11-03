#include "api_tool_config.h"
#include "../logger.h"
#include "../backends/backend.h"
#include "../backends/factory.h"
#include <fstream>
#include <iostream>
#include <iomanip>

json APIToolEntry::to_json() const {
    json j = {
        {"name", name},
        {"backend", backend},
        {"model", model}
    };

    if (!api_key.empty()) {
        j["api_key"] = api_key;
    }

    if (!api_base.empty()) {
        j["api_base"] = api_base;
    }

    if (context_size > 0) {
        j["context_size"] = context_size;
    }

    if (max_tokens > 0) {
        j["max_tokens"] = max_tokens;
    }

    return j;
}

APIToolEntry APIToolEntry::from_json(const json& j) {
    APIToolEntry entry;
    entry.name = j.value("name", "");
    entry.backend = j.value("backend", "");
    entry.model = j.value("model", "");
    entry.api_key = j.value("api_key", "");
    entry.api_base = j.value("api_base", "");
    entry.context_size = j.value("context_size", 0);
    entry.max_tokens = j.value("max_tokens", 0);

    return entry;
}

std::vector<APIToolEntry> APIToolConfig::load(const std::string& config_path) {
    std::vector<APIToolEntry> tools;

    std::ifstream file(config_path);
    if (!file.is_open()) {
        return tools;
    }

    try {
        json config = json::parse(file);

        if (config.contains("api_tools") && config["api_tools"].is_array()) {
            for (const auto& tool_json : config["api_tools"]) {
                tools.push_back(APIToolEntry::from_json(tool_json));
            }
        }
    } catch (const json::exception& e) {
        LOG_ERROR("Failed to parse API tools config: " + std::string(e.what()));
    }

    return tools;
}

bool APIToolConfig::save(const std::string& config_path, const std::vector<APIToolEntry>& tools) {
    std::ifstream infile(config_path);
    json config;

    if (infile.is_open()) {
        try {
            config = json::parse(infile);
        } catch (const json::exception& e) {
            LOG_ERROR("Failed to parse existing config: " + std::string(e.what()));
            config = json::object();
        }
        infile.close();
    } else {
        config = json::object();
    }

    // Update api_tools array
    json api_tools_array = json::array();
    for (const auto& tool : tools) {
        api_tools_array.push_back(tool.to_json());
    }
    config["api_tools"] = api_tools_array;

    // Write back to file
    std::ofstream outfile(config_path);
    if (!outfile.is_open()) {
        LOG_ERROR("Failed to open config file for writing: " + config_path);
        return false;
    }

    outfile << config.dump(2) << std::endl;
    return true;
}

bool APIToolConfig::add_tool(const std::string& config_path, const APIToolEntry& tool) {
    auto tools = load(config_path);

    // Check for duplicates
    for (const auto& existing : tools) {
        if (existing.name == tool.name) {
            std::cerr << "Error: API tool '" << tool.name << "' already exists" << std::endl;
            return false;
        }
    }

    // Validate backend is available
    auto available_backends = BackendFactory::get_available_backends();
    bool backend_valid = false;
    for (const auto& b : available_backends) {
        if (b == tool.backend) {
            backend_valid = true;
            break;
        }
    }

    if (!backend_valid) {
        std::cerr << "Error: Backend '" << tool.backend << "' is not available" << std::endl;
        std::cerr << "Available backends: ";
        for (size_t i = 0; i < available_backends.size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << available_backends[i];
        }
        std::cerr << std::endl;
        return false;
    }

    tools.push_back(tool);
    return save(config_path, tools);
}

bool APIToolConfig::remove_tool(const std::string& config_path, const std::string& name) {
    auto tools = load(config_path);
    bool found = false;

    tools.erase(
        std::remove_if(tools.begin(), tools.end(),
            [&name, &found](const APIToolEntry& t) {
                if (t.name == name) {
                    found = true;
                    return true;
                }
                return false;
            }),
        tools.end()
    );

    if (!found) {
        std::cerr << "Error: API tool '" << name << "' not found" << std::endl;
        return false;
    }

    return save(config_path, tools);
}

void APIToolConfig::list_tools(const std::string& config_path, bool check_health) {
    auto tools = load(config_path);

    if (tools.empty()) {
        std::cout << "No API tools configured" << std::endl;
        return;
    }

    if (check_health) {
        std::cout << "Checking API tool health...\n" << std::endl;
    }

    for (const auto& tool : tools) {
        std::cout << tool.name << ": " << tool.backend << " (" << tool.model << ")";

        if (!tool.api_base.empty()) {
            std::cout << " [custom endpoint]";
        }

        if (check_health) {
            // For health check, we'd need to temporarily create a backend and test it
            // Skip for now to avoid complexity and API calls
            std::cout << " - (health check not implemented)";
        }

        std::cout << std::endl;
    }
}

bool APIToolConfig::tool_exists(const std::string& config_path, const std::string& name) {
    auto tools = load(config_path);
    for (const auto& tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}
