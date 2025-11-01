#include "tool.h"
#include <stdexcept>
#include <algorithm>

ToolRegistry& ToolRegistry::instance() {
    static ToolRegistry registry;
    return registry;
}

void ToolRegistry::register_tool(std::unique_ptr<Tool> tool) {
    if (!tool) {
        throw std::invalid_argument("Tool cannot be null");
    }

    std::string original_name = tool->unsanitized_name();

    // Sanitize to match API backend requirements: ^[a-zA-Z0-9_-]+$
    std::string sanitized_name;
    for (char c : original_name) {
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-') {
            sanitized_name += c;
        } else {
            sanitized_name += '_';  // Replace invalid chars with underscore
        }
    }

    // Convert to lowercase for case-insensitive lookup
    std::transform(sanitized_name.begin(), sanitized_name.end(), sanitized_name.begin(), ::tolower);

    // Store sanitized name in the tool before moving
    tool->set_sanitized_name(sanitized_name);

    tools_[sanitized_name] = std::move(tool);
}

Tool* ToolRegistry::get_tool(const std::string& name) const {
    // Convert to lowercase for case-insensitive lookup
    std::string lowercase_name = name;
    std::transform(lowercase_name.begin(), lowercase_name.end(), lowercase_name.begin(), ::tolower);

    auto it = tools_.find(lowercase_name);
    if (it != tools_.end()) {
        return it->second.get();
    }
    return nullptr;
}

std::vector<std::string> ToolRegistry::list_tools() const {
    std::vector<std::string> tool_names;
    for (const auto& pair : tools_) {
        tool_names.push_back(pair.first);
    }
    return tool_names;
}

std::map<std::string, std::string> ToolRegistry::list_tools_with_descriptions() const {
    std::map<std::string, std::string> tools_with_descriptions;
    for (const auto& pair : tools_) {
        tools_with_descriptions[pair.first] = pair.second->description();
    }
    return tools_with_descriptions;
}

std::string ToolRegistry::get_tools_as_system_prompt() const {
    std::string prompt = "Here are the available tools:\n\n";

    auto tool_descriptions = list_tools_with_descriptions();

    // Sort tools so memory tools appear first
    std::vector<std::string> memory_tools = {"search_memory", "get_fact", "set_fact", "clear_fact"};
    std::vector<std::string> other_tools;

    // Separate memory tools from other tools
    for (const auto& pair : tool_descriptions) {
        if (std::find(memory_tools.begin(), memory_tools.end(), pair.first) == memory_tools.end()) {
            other_tools.push_back(pair.first);
        }
    }

    // Add memory tools first
    for (const auto& tool_name : memory_tools) {
        if (tool_descriptions.find(tool_name) != tool_descriptions.end()) {
            Tool* tool = get_tool(tool_name);
            if (tool) {
                prompt += "- " + tool_name + ": " + tool_descriptions[tool_name] + " (parameters: " + tool->parameters() + ")\n";
            }
        }
    }

    // Add other tools
    for (const auto& tool_name : other_tools) {
        Tool* tool = get_tool(tool_name);
        if (tool) {
            prompt += "- " + tool_name + ": " + tool_descriptions[tool_name] + " (parameters: " + tool->parameters() + ")\n";
        }
    }

    // Add tool call format instructions
    prompt += "\n**IMPORTANT - Tool Call Format:**\n";
    prompt += "Tool calls must be raw JSON as the ENTIRE message (no markdown, no text, no quotes).\n";
    prompt += "Format: {\"name\": \"tool_name\", \"parameters\": {\"key\": \"value\"}}\n";
    prompt += "Use \"name\" (not \"tool_name\") and \"parameters\" (not \"params\").\n";
    prompt += "If a tool fails, continue gracefully without retrying indefinitely.\n";

    return prompt;
}

ToolResult execute_tool(const std::string& tool_name,
                       const std::map<std::string, std::any>& parameters) {
    // Get tool from registry
    Tool* tool = ToolRegistry::instance().get_tool(tool_name);

    if (!tool) {
        return ToolResult(false, "", "Tool not found: " + tool_name);
    }

    try {
        // Execute tool
        auto result = tool->execute(parameters);

        // Convert result map to string
        // Tools can return {"content": "..."} or {"output": "..."} or {"error": "..."}
        std::string output;

        if (result.find("error") != result.end()) {
            try {
                output = std::any_cast<std::string>(result["error"]);
                return ToolResult(false, "", output);
            } catch (const std::bad_any_cast&) {
                return ToolResult(false, "", "Tool returned error but couldn't cast to string");
            }
        }

        // Check for "content" key (used by most tools)
        if (result.find("content") != result.end()) {
            try {
                output = std::any_cast<std::string>(result["content"]);
                return ToolResult(true, output);
            } catch (const std::bad_any_cast&) {
                return ToolResult(false, "", "Tool returned content but couldn't cast to string");
            }
        }

        // Check for "output" key (legacy/alternative)
        if (result.find("output") != result.end()) {
            try {
                output = std::any_cast<std::string>(result["output"]);
                return ToolResult(true, output);
            } catch (const std::bad_any_cast&) {
                return ToolResult(false, "", "Tool returned output but couldn't cast to string");
            }
        }

        // Fallback: tool returned something else
        return ToolResult(false, "", "Tool returned unexpected result format");

    } catch (const std::exception& e) {
        return ToolResult(false, "", std::string("Tool execution failed: ") + e.what());
    } catch (...) {
        return ToolResult(false, "", "Tool execution failed with unknown error");
    }
}

namespace tool_utils {
    std::string get_string(const std::map<std::string, std::any>& args, const std::string& key, const std::string& default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<std::string>(it->second);
            } catch (const std::bad_any_cast&) {
                // Try to convert from const char*
                try {
                    return std::string(std::any_cast<const char*>(it->second));
                } catch (const std::bad_any_cast&) {
                    return default_value;
                }
            }
        }
        return default_value;
    }

    int get_int(const std::map<std::string, std::any>& args, const std::string& key, int default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<int>(it->second);
            } catch (const std::bad_any_cast&) {
                try {
                    return static_cast<int>(std::any_cast<double>(it->second));
                } catch (const std::bad_any_cast&) {
                    return default_value;
                }
            }
        }
        return default_value;
    }

    double get_double(const std::map<std::string, std::any>& args, const std::string& key, double default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<double>(it->second);
            } catch (const std::bad_any_cast&) {
                try {
                    return static_cast<double>(std::any_cast<int>(it->second));
                } catch (const std::bad_any_cast&) {
                    return default_value;
                }
            }
        }
        return default_value;
    }

    bool get_bool(const std::map<std::string, std::any>& args, const std::string& key, bool default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<bool>(it->second);
            } catch (const std::bad_any_cast&) {
                return default_value;
            }
        }
        return default_value;
    }
}