#pragma once

#include <string>
#include <map>
#include <memory>
#include <functional>
#include <any>
#include <vector>

// Parameter definition for JSON schema generation
struct ParameterDef {
    std::string name;
    std::string type;        // "string", "number", "boolean", "array", "object"
    std::string description;
    bool required;
    std::string default_value;  // Optional default value as string

    // For array types
    std::string array_item_type;  // e.g., "string", "object"

    // For object types (nested parameters)
    std::vector<ParameterDef> object_properties;
};

// Base tool interface
class Tool {
public:
    virtual ~Tool() = default;

    // Tool implementations must provide unsanitized name (may contain colons, etc.)
    virtual std::string unsanitized_name() const = 0;
    virtual std::string description() const = 0;
    virtual std::string parameters() const = 0;  // Legacy string format for backward compatibility
    virtual std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) = 0;

    /// @brief Get structured parameter definitions for JSON schema generation
    /// @return Vector of parameter definitions (empty vector = not yet implemented, will fall back to legacy parameters())
    virtual std::vector<ParameterDef> get_parameters_schema() const { return {}; }

    // The actual sanitized name to use (set during registration)
    std::string name() const { return sanitized_name; }
    void set_sanitized_name(const std::string& name) { sanitized_name = name; }

    std::string sanitized_name;
};

// Tool execution result
struct ToolResult {
    bool success;
    std::string content;   // Full data sent to model
    std::string summary;   // One-line summary for display
    std::string error;

    ToolResult() : success(false) {}
    ToolResult(bool s, const std::string& c, const std::string& e = "")
        : success(s), content(c), error(e) {}
    ToolResult(bool s, const std::string& c, const std::string& sum, const std::string& e)
        : success(s), content(c), summary(sum), error(e) {}
};

// Utility functions for std::any conversion
namespace tool_utils {
    std::string get_string(const std::map<std::string, std::any>& args, const std::string& key, const std::string& default_value = "");
    int get_int(const std::map<std::string, std::any>& args, const std::string& key, int default_value = 0);
    double get_double(const std::map<std::string, std::any>& args, const std::string& key, double default_value = 0.0);
    bool get_bool(const std::map<std::string, std::any>& args, const std::string& key, bool default_value = false);
}
