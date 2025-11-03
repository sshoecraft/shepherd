#pragma once

#include "shepherd.h"
#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <map>

/// @brief API tool configuration entry
struct APIToolEntry {
    std::string name;
    std::string backend;
    std::string model;
    std::string api_key;
    std::string api_base;
    size_t context_size;
    int max_tokens;

    nlohmann::json to_json() const;
    static APIToolEntry from_json(const nlohmann::json& j);
};

/// @brief API tool configuration manager
class APIToolConfig {
public:
    /// @brief Load API tools from config file
    static std::vector<APIToolEntry> load(const std::string& config_path);

    /// @brief Save API tools to config file
    static bool save(const std::string& config_path, const std::vector<APIToolEntry>& tools);

    /// @brief Add a new API tool
    static bool add_tool(const std::string& config_path, const APIToolEntry& tool);

    /// @brief Remove an API tool by name
    static bool remove_tool(const std::string& config_path, const std::string& name);

    /// @brief List all API tools
    /// @param check_health If true, test connection to each backend
    static void list_tools(const std::string& config_path, bool check_health = false);

    /// @brief Check if tool exists
    static bool tool_exists(const std::string& config_path, const std::string& name);
};
