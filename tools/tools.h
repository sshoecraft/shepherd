#pragma once

#include "tool.h"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <functional>

// Forward declaration
class Session;

/// @brief Manages all tools by category with unified access
class Tools {
public:
    // Storage by category (owns the tools)
    std::vector<std::unique_ptr<Tool>> core_tools;
    std::vector<std::unique_ptr<Tool>> mcp_tools;
    std::vector<std::unique_ptr<Tool>> api_tools;

    // Flat list of raw pointers for fast iteration (rebuilt by build_all_tools)
    std::vector<Tool*> all_tools;

    // Name-based lookup (rebuilt by build_all_tools)
    std::map<std::string, Tool*> by_name;

    // Enable/disable state
    std::map<std::string, bool> enabled;

    // Register a tool to a specific category
    void register_tool(std::unique_ptr<Tool> tool, const std::string& category = "core");

    // Rebuild all_tools and by_name from category vectors
    void build_all_tools();

    // Lookup tool by name
    Tool* get(const std::string& name);

    // List all tool names
    std::vector<std::string> list();

    // List tools with descriptions
    std::map<std::string, std::string> list_with_descriptions();

    // Format tools for system prompt (local backends)
    std::string as_system_prompt();

    // Enable/disable tools (supports multiple tools)
    void enable(const std::vector<std::string>& names);
    void disable(const std::vector<std::string>& names);
    void enable(const std::string& name);
    void disable(const std::string& name);
    bool is_enabled(const std::string& name);

    // Command-line and slash command handler
    // Returns 0 on success, non-zero on error
    // callback: function to emit output
    int handle_tools_args(const std::vector<std::string>& args,
                          std::function<void(const std::string&)> callback);

    // Execute a tool by name with JSON parameters string
    // Returns ToolResult with success/failure, content, and error
    ToolResult execute(const std::string& tool_name, const std::string& params_json);

    // Populate session.tools from all_tools
    void populate_session_tools(Session& session);

    // Clear all tools in a category
    void clear_category(const std::string& category);

    // Remove a tool by name
    void remove_tool(const std::string& name);

    // Check if native (core) tools are registered
    bool has_native_tools() const { return !core_tools.empty(); }
};

// Include individual tool headers for direct access if needed
#include "memory_tools.h"
#include "filesystem_tools.h"
#include "command_tools.h"
#include "json_tools.h"
#include "http_tools.h"
#include "mcp_resource_tools.h"
#include "core_tools.h"
