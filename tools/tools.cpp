#include "tools.h"
#include "../session.h"
#include "../logger.h"
#include <algorithm>
#include <sstream>

// Helper: sanitize tool name to match API backend requirements: ^[a-zA-Z0-9_-]+$
static std::string sanitize_tool_name(const std::string& name) {
    std::string sanitized;
    for (char c : name) {
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-') {
            sanitized += c;
        } else {
            sanitized += '_';
        }
    }
    // Convert to lowercase
    std::transform(sanitized.begin(), sanitized.end(), sanitized.begin(), ::tolower);
    return sanitized;
}

void Tools::register_tool(std::unique_ptr<Tool> tool, const std::string& category) {
    if (!tool) return;

    // Sanitize and set the tool name
    std::string sanitized = sanitize_tool_name(tool->unsanitized_name());
    tool->set_sanitized_name(sanitized);

    if (category == "mcp") {
        mcp_tools.push_back(std::move(tool));
    } else if (category == "api") {
        api_tools.push_back(std::move(tool));
    } else {
        core_tools.push_back(std::move(tool));
    }
}

void Tools::build_all_tools() {
    all_tools.clear();
    by_name.clear();

    // Add core tools
    for (auto& tool : core_tools) {
        all_tools.push_back(tool.get());
        by_name[tool->name()] = tool.get();
    }

    // Add MCP tools
    for (auto& tool : mcp_tools) {
        all_tools.push_back(tool.get());
        by_name[tool->name()] = tool.get();
    }

    // Add API tools
    for (auto& tool : api_tools) {
        all_tools.push_back(tool.get());
        by_name[tool->name()] = tool.get();
    }

    LOG_DEBUG("Tools::build_all_tools() - " + std::to_string(all_tools.size()) + " tools total");
}

Tool* Tools::get(const std::string& name) {
    // Convert to lowercase for case-insensitive lookup
    std::string lowercase_name = name;
    std::transform(lowercase_name.begin(), lowercase_name.end(), lowercase_name.begin(), ::tolower);

    auto it = by_name.find(lowercase_name);
    if (it != by_name.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::string> Tools::list() {
    std::vector<std::string> names;
    for (Tool* tool : all_tools) {
        names.push_back(tool->name());
    }
    return names;
}

std::map<std::string, std::string> Tools::list_with_descriptions() {
    std::map<std::string, std::string> result;
    for (Tool* tool : all_tools) {
        result[tool->name()] = tool->description();
    }
    return result;
}

std::string Tools::as_system_prompt() {
    std::ostringstream prompt;
    prompt << "Here are the available tools:\n\n";

    // Sort tools so memory tools appear first
    std::vector<std::string> memory_tool_names = {"search_memory", "get_fact", "set_fact", "clear_fact"};
    std::vector<Tool*> memory_tools_list;
    std::vector<Tool*> other_tools_list;

    for (Tool* tool : all_tools) {
        if (!is_enabled(tool->name())) continue;

        bool is_memory_tool = std::find(memory_tool_names.begin(), memory_tool_names.end(),
                                        tool->name()) != memory_tool_names.end();
        if (is_memory_tool) {
            memory_tools_list.push_back(tool);
        } else {
            other_tools_list.push_back(tool);
        }
    }

    // Add memory tools first
    for (Tool* tool : memory_tools_list) {
        prompt << "- " << tool->name() << ": " << tool->description()
               << " (parameters: " << tool->parameters() << ")\n";
    }

    // Add other tools
    for (Tool* tool : other_tools_list) {
        prompt << "- " << tool->name() << ": " << tool->description()
               << " (parameters: " << tool->parameters() << ")\n";
    }

    // Add tool call format instructions
    prompt << "\n**IMPORTANT - Tool Call Format:**\n";
    prompt << "Tool calls must be raw JSON as the ENTIRE message (no markdown, no text, no quotes).\n";
    prompt << "Format: {\"name\": \"tool_name\", \"parameters\": {\"key\": \"value\"}}\n";
    prompt << "Use \"name\" (not \"tool_name\") and \"parameters\" (not \"params\").\n";
    prompt << "If a tool fails, continue gracefully without retrying indefinitely.\n";

    return prompt.str();
}

void Tools::enable(const std::string& name) {
    std::string lowercase_name = name;
    std::transform(lowercase_name.begin(), lowercase_name.end(), lowercase_name.begin(), ::tolower);
    enabled[lowercase_name] = true;
}

void Tools::enable(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        enable(name);
    }
}

void Tools::disable(const std::string& name) {
    std::string lowercase_name = name;
    std::transform(lowercase_name.begin(), lowercase_name.end(), lowercase_name.begin(), ::tolower);
    enabled[lowercase_name] = false;
}

void Tools::disable(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        disable(name);
    }
}

bool Tools::is_enabled(const std::string& name) {
    std::string lowercase_name = name;
    std::transform(lowercase_name.begin(), lowercase_name.end(), lowercase_name.begin(), ::tolower);

    auto it = enabled.find(lowercase_name);
    if (it != enabled.end()) {
        return it->second;
    }
    // Default to enabled
    return true;
}

int Tools::handle_tools_args(const std::vector<std::string>& args) {
    // No args or "list" - list all tools
    if (args.empty() || args[0] == "list") {
        auto tool_descriptions = list_with_descriptions();

        printf("\n=== Available Tools (%zu) ===\n\n", tool_descriptions.size());

        size_t enabled_count = 0;
        size_t disabled_count = 0;

        for (const auto& pair : tool_descriptions) {
            bool tool_enabled = is_enabled(pair.first);
            if (tool_enabled) {
                enabled_count++;
            } else {
                disabled_count++;
            }

            const char* status = tool_enabled ? "[enabled]" : "[DISABLED]";
            printf("  %s %s\n", status, pair.first.c_str());
            printf("    %s\n\n", pair.second.c_str());
        }

        printf("Total: %zu tools (%zu enabled, %zu disabled)\n",
               tool_descriptions.size(), enabled_count, disabled_count);
        return 0;
    }

    // "enable <tool1> [tool2] ..." - enable one or more tools
    if (args[0] == "enable") {
        if (args.size() < 2) {
            printf("Usage: tools enable <tool_name> [tool_name2] ...\n");
            return 1;
        }

        std::vector<std::string> not_found;
        std::vector<std::string> enabled_tools;

        for (size_t i = 1; i < args.size(); i++) {
            Tool* tool = get(args[i]);
            if (!tool) {
                not_found.push_back(args[i]);
            } else {
                enable(args[i]);
                enabled_tools.push_back(tool->name());
            }
        }

        if (!enabled_tools.empty()) {
            printf("Enabled: ");
            for (size_t i = 0; i < enabled_tools.size(); i++) {
                if (i > 0) printf(", ");
                printf("%s", enabled_tools[i].c_str());
            }
            printf("\n");
        }

        if (!not_found.empty()) {
            printf("Not found: ");
            for (size_t i = 0; i < not_found.size(); i++) {
                if (i > 0) printf(", ");
                printf("%s", not_found[i].c_str());
            }
            printf("\n");
            return 1;
        }

        return 0;
    }

    // "disable <tool1> [tool2] ..." - disable one or more tools
    if (args[0] == "disable") {
        if (args.size() < 2) {
            printf("Usage: tools disable <tool_name> [tool_name2] ...\n");
            return 1;
        }

        std::vector<std::string> not_found;
        std::vector<std::string> disabled_tools;

        for (size_t i = 1; i < args.size(); i++) {
            Tool* tool = get(args[i]);
            if (!tool) {
                not_found.push_back(args[i]);
            } else {
                disable(args[i]);
                disabled_tools.push_back(tool->name());
            }
        }

        if (!disabled_tools.empty()) {
            printf("Disabled: ");
            for (size_t i = 0; i < disabled_tools.size(); i++) {
                if (i > 0) printf(", ");
                printf("%s", disabled_tools[i].c_str());
            }
            printf("\n");
        }

        if (!not_found.empty()) {
            printf("Not found: ");
            for (size_t i = 0; i < not_found.size(); i++) {
                if (i > 0) printf(", ");
                printf("%s", not_found[i].c_str());
            }
            printf("\n");
            return 1;
        }

        return 0;
    }

    // Unknown subcommand
    printf("Usage: tools [list | enable <tool_name>... | disable <tool_name>...]\n");
    return 1;
}

ToolResult Tools::execute(const std::string& tool_name, const std::map<std::string, std::any>& parameters) {
    Tool* tool = get(tool_name);

    if (!tool) {
        return ToolResult(false, "", "Tool not found: " + tool_name);
    }

    if (!is_enabled(tool_name)) {
        return ToolResult(false, "", "Tool is disabled: " + tool_name);
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

        // No recognized output key - return empty success
        return ToolResult(true, "");

    } catch (const std::exception& e) {
        return ToolResult(false, "", std::string("Tool execution failed: ") + e.what());
    }
}

void Tools::populate_session_tools(Session& session) {
    session.tools.clear();

    for (Tool* tool : all_tools) {
        if (!is_enabled(tool->name())) continue;

        Session::Tool st;
        st.name = tool->name();
        st.description = tool->description();

        // Build JSON schema from parameter definitions
        auto params = tool->get_parameters_schema();
        nlohmann::json schema;
        schema["type"] = "object";
        schema["properties"] = nlohmann::json::object();
        nlohmann::json required_arr = nlohmann::json::array();

        for (const auto& param : params) {
            nlohmann::json param_schema;
            param_schema["type"] = param.type;
            if (!param.description.empty()) {
                param_schema["description"] = param.description;
            }
            schema["properties"][param.name] = param_schema;

            if (param.required) {
                required_arr.push_back(param.name);
            }
        }

        if (!required_arr.empty()) {
            schema["required"] = required_arr;
        }

        st.parameters = schema;
        session.tools.push_back(st);
    }

    LOG_DEBUG("Populated session with " + std::to_string(session.tools.size()) + " tools");
}

void Tools::clear_category(const std::string& category) {
    if (category == "mcp") {
        mcp_tools.clear();
    } else if (category == "api") {
        api_tools.clear();
    } else if (category == "core") {
        core_tools.clear();
    } else if (category == "all") {
        core_tools.clear();
        mcp_tools.clear();
        api_tools.clear();
    }
    build_all_tools();
}

void Tools::remove_tool(const std::string& name) {
    std::string lowercase_name = name;
    std::transform(lowercase_name.begin(), lowercase_name.end(), lowercase_name.begin(), ::tolower);

    // Remove from all category vectors (unique_ptr version)
    auto remove_from_vec = [&lowercase_name](std::vector<std::unique_ptr<Tool>>& vec) {
        vec.erase(std::remove_if(vec.begin(), vec.end(),
            [&lowercase_name](const std::unique_ptr<Tool>& t) {
                return t->name() == lowercase_name;
            }),
            vec.end());
    };

    remove_from_vec(core_tools);
    remove_from_vec(mcp_tools);
    remove_from_vec(api_tools);

    // Rebuild all_tools and by_name
    build_all_tools();
}
