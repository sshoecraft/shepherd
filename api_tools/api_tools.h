#pragma once

#include "shepherd.h"
#include "api_tool_config.h"
#include "api_tool_adapter.h"
#include "../tools/tool.h"
#include <vector>
#include <memory>
#include <string>

/// @brief Manages API tools and registers them with Shepherd's ToolRegistry
class APITools {
public:
    static APITools& instance();

    /// @brief Initialize API tools from config
    /// @return True if at least one tool initialized successfully
    bool initialize();

    /// @brief Initialize from explicit tool configs (for testing)
    /// @param tool_configs List of tool configurations
    /// @return True if at least one tool initialized successfully
    bool initialize(const std::vector<APIToolEntry>& tool_configs);

    /// @brief Shutdown all API tools
    void shutdown();

    /// @brief Get total number of API tools registered
    size_t get_tool_count() const { return total_tools; }

    /// @brief Get list of tool names
    std::vector<std::string> get_tool_names() const;

private:
    APITools() = default;
    ~APITools() { shutdown(); }

    // Disable copy
    APITools(const APITools&) = delete;
    APITools& operator=(const APITools&) = delete;

    /// @brief Register a single API tool
    /// @param config Tool configuration
    /// @return True if successful
    bool register_tool(const APIToolEntry& config);

    size_t total_tools = 0;
};
