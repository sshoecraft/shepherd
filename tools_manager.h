#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <any>
#include <stdexcept>

/// @brief Result of a tool execution
struct ToolResult {
    bool success;
    std::string content;
    std::string error_message;
    std::map<std::string, std::any> metadata;

    ToolResult(bool success = false, const std::string& content = "", const std::string& error = "")
        : success(success), content(content), error_message(error) {}
};

/// @brief Information about an available tool
struct ToolInfo {
    std::string name;
    std::string description;
    std::string parameters_schema;  // JSON schema or backend-specific format

    ToolInfo(const std::string& name = "", const std::string& desc = "", const std::string& params = "")
        : name(name), description(desc), parameters_schema(params) {}
};

/// @brief Abstract base class for tool management
/// Provides unified interface for all backends to access and execute tools
class ToolsManager {
public:
    virtual ~ToolsManager() = default;

    /// @brief Get list of all available tools
    /// @return Vector of tool information
    virtual std::vector<ToolInfo> get_available_tools() const = 0;

    /// @brief Get specific tool information
    /// @param tool_name Name of the tool
    /// @return Tool information, or empty ToolInfo if not found
    virtual ToolInfo get_tool_info(const std::string& tool_name) const = 0;

    /// @brief Check if a tool is available
    /// @param tool_name Name of the tool
    /// @return True if tool exists and is available
    virtual bool has_tool(const std::string& tool_name) const = 0;

    /// @brief Execute a tool with given arguments
    /// @param tool_name Name of the tool to execute
    /// @param arguments Tool arguments as key-value pairs
    /// @return Result of tool execution
    virtual ToolResult execute_tool(const std::string& tool_name,
                                   const std::map<std::string, std::any>& arguments) = 0;

    /// @brief Get tools formatted for backend-specific context
    /// Each backend implements this to format tools for their LLM
    /// @return Backend-specific representation of available tools
    virtual std::string get_tools_for_context() = 0;

    /// @brief Parse tool calls from LLM response
    /// Each backend implements this to extract tool calls from their LLM's response format
    /// @param llm_response Raw response from the LLM
    /// @return Vector of parsed tool calls (tool_name, arguments pairs)
    virtual std::vector<std::pair<std::string, std::map<std::string, std::any>>>
           parse_tool_calls(const std::string& llm_response) = 0;

    /// @brief Format tool results for inclusion in context
    /// Each backend implements this to format tool results for their LLM
    /// @param tool_name Name of the executed tool
    /// @param result Result of tool execution
    /// @return Backend-specific formatted result
    virtual std::string format_tool_result(const std::string& tool_name,
                                          const ToolResult& result) = 0;

protected:
    /// @brief Reference to the global tool registry
    /// Set during initialization in main.cpp
    static void* tool_registry_;
};

/// @brief Exception thrown by tools manager
class ToolsManagerError : public std::runtime_error {
public:
    explicit ToolsManagerError(const std::string& message)
        : std::runtime_error("ToolsManager: " + message) {}
};