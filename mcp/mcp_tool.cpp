#include "shepherd.h"
#include "mcp_tool.h"
#include <sstream>


MCPToolAdapter::MCPToolAdapter(std::shared_ptr<MCPClient> cli, const MCPTool& tool)
    : client(cli), mcp_tool(tool) {
}

std::string MCPToolAdapter::unsanitized_name() const {
    // Prefix with server name to avoid conflicts
    return client->server->server_config.name + ":" + mcp_tool.name;
}

std::string MCPToolAdapter::description() const {
    return mcp_tool.description;
}

std::string MCPToolAdapter::parameters() const {
    // Return the JSON schema directly for API backends
    // API backends need the actual JSON schema, not human-readable format
    return mcp_tool.input_schema.dump();
}

std::vector<ParameterDef> MCPToolAdapter::get_parameters_schema() const {
    std::vector<ParameterDef> params;

    if (!mcp_tool.input_schema.contains("properties")) {
        return params;
    }

    // Parse required fields list
    std::vector<std::string> required_fields;
    if (mcp_tool.input_schema.contains("required") && mcp_tool.input_schema["required"].is_array()) {
        for (const auto& req : mcp_tool.input_schema["required"]) {
            if (req.is_string()) {
                required_fields.push_back(req.get<std::string>());
            }
        }
    }

    // Convert each property to ParameterDef
    for (auto it = mcp_tool.input_schema["properties"].begin();
         it != mcp_tool.input_schema["properties"].end(); ++it) {

        std::string param_name = it.key();
        auto param_schema = it.value();

        ParameterDef param;
        param.name = param_name;

        // Get type (default to string if not specified)
        param.type = "string";
        if (param_schema.contains("type") && param_schema["type"].is_string()) {
            param.type = param_schema["type"].get<std::string>();
        }

        // Get description
        if (param_schema.contains("description") && param_schema["description"].is_string()) {
            param.description = param_schema["description"].get<std::string>();
        }

        // Check if required
        param.required = std::find(required_fields.begin(), required_fields.end(), param_name) != required_fields.end();

        params.push_back(param);
    }

    return params;
}

std::map<std::string, std::any> MCPToolAdapter::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    try {
        // Convert Shepherd args to MCP JSON
        nlohmann::json mcp_args = args_to_json(args);

        dout(1) << "Executing MCP tool: " + mcp_tool.name << std::endl;

        // Call MCP tool
        nlohmann::json mcp_result = client->call_tool(mcp_tool.name, mcp_args);

        // Extract content from MCP result
        if (mcp_result.contains("content") && mcp_result["content"].is_array() &&
            !mcp_result["content"].empty()) {

            auto content = mcp_result["content"][0];
            if (content.contains("text")) {
                std::string text_content = content["text"].get<std::string>();

                // Hard limit on MCP tool results: max 50K characters
                const size_t MAX_MCP_RESULT_CHARS = 50000;
                if (text_content.length() > MAX_MCP_RESULT_CHARS) {
                    size_t original_length = text_content.length();
                    size_t original_lines = std::count(text_content.begin(), text_content.end(), '\n');

                    // Truncate to limit
                    text_content = text_content.substr(0, MAX_MCP_RESULT_CHARS);
                    size_t truncated_lines = std::count(text_content.begin(), text_content.end(), '\n');

                    // Add truncation notice
                    text_content += "\n\n[TRUNCATED: MCP tool result exceeded " +
                                   std::to_string(MAX_MCP_RESULT_CHARS) + " character limit]\n";
                    text_content += "Original size: " + std::to_string(original_length) +
                                   " characters (" + std::to_string(original_lines) + " lines)\n";
                    text_content += "Consider using pagination, filters, or reading specific sections.";

                    dout(1) << "WARNING: Truncated MCP tool result from " + mcp_tool.name +
                            " (" + std::to_string(original_length) + " chars -> " +
                            std::to_string(MAX_MCP_RESULT_CHARS) + " chars)" << std::endl;

                    result["success"] = true;
                    result["content"] = text_content;
                    result["summary"] = "Truncated: " + std::to_string(truncated_lines) + "/" +
                                       std::to_string(original_lines) + " lines";
                } else {
                    result["success"] = true;
                    result["content"] = text_content;
                }
            } else {
                result["success"] = false;
                result["error"] = std::string("No text content in MCP response");
            }
        } else if (mcp_result.contains("isError") && mcp_result["isError"].get<bool>()) {
            result["success"] = false;
            result["error"] = mcp_result.value("content", "Unknown error");
        } else {
            result["success"] = true;
            result["content"] = mcp_result.dump();
        }

    } catch (const std::exception& e) {
        std::cerr << "MCP tool execution failed: " + std::string(e.what()) << std::endl;
        result["success"] = false;
        result["error"] = std::string("MCP error: ") + e.what();
    }

    return result;
}

std::string MCPToolAdapter::schema_to_parameters(const nlohmann::json& schema) const {
    if (!schema.contains("properties")) {
        return "";
    }

    std::ostringstream oss;
    bool first = true;

    for (auto it = schema["properties"].begin(); it != schema["properties"].end(); ++it) {
        if (!first) oss << ", ";
        first = false;

        std::string param_name = it.key();
        auto param_schema = it.value();

        oss << param_name << "=\"";

        // Add type hint
        if (param_schema.contains("type")) {
            oss << param_schema["type"].get<std::string>();
        }

        // Add description
        if (param_schema.contains("description")) {
            oss << " - " << param_schema["description"].get<std::string>();
        }

        oss << "\"";
    }

    return oss.str();
}

nlohmann::json MCPToolAdapter::args_to_json(const std::map<std::string, std::any>& args) const {
    nlohmann::json result = nlohmann::json::object();

    for (const auto& [key, value] : args) {
        if (value.type() == typeid(std::string)) {
            result[key] = std::any_cast<std::string>(value);
        } else if (value.type() == typeid(int)) {
            result[key] = std::any_cast<int>(value);
        } else if (value.type() == typeid(double)) {
            result[key] = std::any_cast<double>(value);
        } else if (value.type() == typeid(bool)) {
            result[key] = std::any_cast<bool>(value);
        } else if (value.type() == typeid(const char*)) {
            result[key] = std::string(std::any_cast<const char*>(value));
        } else {
            dout(1) << "Unknown parameter type for: " + key << std::endl;
        }
    }

    return result;
}
