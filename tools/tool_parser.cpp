#include "tool_parser.h"
#include "../logger.h"
#include "../nlohmann/json.hpp"
#include <sstream>
#include <regex>

namespace ToolParser {

// Forward declaration
std::optional<ToolCall> parse_xml_function_block(const std::string& xml_content);

std::string extract_json(const std::string& response) {
    // Find the first { and track braces to find the matching }
    size_t json_start = response.find('{');
    if (json_start == std::string::npos) {
        return "";
    }

    int brace_count = 0;
    size_t json_end = json_start;

    for (size_t i = json_start; i < response.length(); i++) {
        if (response[i] == '{') {
            brace_count++;
        } else if (response[i] == '}') {
            brace_count--;
        }

        if (brace_count == 0) {
            json_end = i + 1;
            break;
        }
    }

    if (brace_count != 0) {
        // Unmatched braces
        return "";
    }

    return response.substr(json_start, json_end - json_start);
}

bool has_tool_call(const std::string& response,
                  const std::vector<std::string>& tool_call_markers) {
    // Tool call detection: scan entire response for valid tool call JSON
    // Changed from requiring "JSON on its own line" to support models that
    // output explanatory text before tool calls (e.g., GLM-4.5)

    // Try to extract and parse JSON from anywhere in the response
    std::string json_str = extract_json(response);
    if (json_str.empty()) {
        return false;
    }

    // Quick sanity check: tool calls should contain "name" somewhere in the JSON
    // This avoids expensive parsing of JSON-like structures that clearly aren't tool calls
    if (json_str.find("\"name\"") == std::string::npos && json_str.find("'name'") == std::string::npos) {
        return false;
    }

    try {
        auto json_response = nlohmann::json::parse(json_str);
        // Accept both "parameters" (internal format) and "arguments" (OpenAI format)
        if (json_response.contains("name") &&
            (json_response.contains("parameters") || json_response.contains("arguments"))) {
            LOG_DEBUG("Found valid tool call");
            return true;
        }
    } catch (const nlohmann::json::exception& e) {
        // Silently ignore parse errors - many false positives from text containing braces
        return false;
    }

    return false;
}

std::optional<ToolCall> parse_xml_tool_call(const std::string& response) {
    // Parse XML-style tool calls (Qwen/Unsloth format):
    // Standard: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    // Lenient: Also accept <function=name>...</function> without outer <tool_call> tags

    size_t tool_call_start = response.find("<tool_call>");
    size_t function_start = response.find("<function=");

    LOG_DEBUG("XML tool call search: tool_call_start=" + std::to_string(tool_call_start) +
              ", function_start=" + std::to_string(function_start));

    // Prioritize whichever tag appears first in the response
    // This handles cases where model generates valid <function=...> but then hallucinates garbage with <tool_call>
    bool use_tool_call_wrapper = (tool_call_start != std::string::npos) &&
                                  (function_start == std::string::npos || tool_call_start < function_start);

    // If we have <tool_call> AND it comes before <function=, try that first
    if (use_tool_call_wrapper) {
        size_t tool_call_end = response.find("</tool_call>", tool_call_start);
        if (tool_call_end == std::string::npos) {
            LOG_DEBUG("Found <tool_call> but no closing </tool_call>, trying <function= fallback");
            // Fall through to try <function= without wrapper
        } else {
            std::string tool_call_content = response.substr(tool_call_start, tool_call_end - tool_call_start + 12);

            // Try parsing arg_key/arg_value format first (GLM-4 format)
            // Format: <tool_call>function_name\n<arg_key>key</arg_key>\n<arg_value>value</arg_value>\n</tool_call>
            std::regex tool_name_regex("<tool_call>\\s*([a-zA-Z_][a-zA-Z0-9_]*)");
            std::smatch tool_name_match;
            if (std::regex_search(tool_call_content, tool_name_match, tool_name_regex)) {
                std::string tool_name = tool_name_match[1].str();
                LOG_DEBUG("Found tool name in <tool_call>: " + tool_name);

                // Extract arg_key/arg_value pairs
                std::map<std::string, std::any> tool_params;
                std::regex arg_regex("<arg_key>\\s*([^<]+)\\s*</arg_key>\\s*<arg_value>\\s*([\\s\\S]*?)\\s*</arg_value>");
                std::sregex_iterator arg_it(tool_call_content.begin(), tool_call_content.end(), arg_regex);
                std::sregex_iterator arg_end;

                for (; arg_it != arg_end; ++arg_it) {
                    std::string key = (*arg_it)[1].str();
                    std::string value = (*arg_it)[2].str();

                    // Trim whitespace
                    size_t start = key.find_first_not_of(" \t\n\r");
                    size_t end = key.find_last_not_of(" \t\n\r");
                    if (start != std::string::npos && end != std::string::npos) {
                        key = key.substr(start, end - start + 1);
                    }

                    start = value.find_first_not_of(" \t\n\r");
                    end = value.find_last_not_of(" \t\n\r");
                    if (start != std::string::npos && end != std::string::npos) {
                        value = value.substr(start, end - start + 1);
                    }

                    tool_params[key] = value;
                    LOG_DEBUG("  Parameter: " + key + " = " + value.substr(0, 50) + (value.length() > 50 ? "..." : ""));
                }

                // Build JSON from parameters for raw_json field
                nlohmann::json params_json;
                for (const auto& [key, value] : tool_params) {
                    params_json[key] = std::any_cast<std::string>(value);
                }

                ToolCall tc;
                tc.name = tool_name;
                tc.parameters = tool_params;
                tc.raw_json = params_json.dump();
                tc.tool_call_id = "";  // GLM models don't provide call IDs
                return tc;
            }

            // Fall back to old <function= format
            auto result = parse_xml_function_block(tool_call_content);
            if (result.has_value()) {
                return result;
            }
            LOG_DEBUG("Failed to parse <tool_call> content, trying <function= fallback");
            // Fall through to try <function= without wrapper
        }
    }

    // Lenient mode: accept <function=name> without outer <tool_call> wrapper
    if (function_start != std::string::npos) {
        size_t function_end = response.find("</function>", function_start);
        LOG_DEBUG("Lenient mode: function_end=" + std::to_string(function_end));
        if (function_end == std::string::npos) {
            LOG_DEBUG("Found <function= but no closing </function>");
            // Show a snippet to help debug
            std::string snippet = response.substr(function_start, std::min(size_t(200), response.length() - function_start));
            LOG_DEBUG("Response snippet: " + snippet);
            return std::nullopt;
        }
        std::string function_content = response.substr(function_start, function_end - function_start + 11);
        LOG_DEBUG("Found function block without <tool_call> wrapper (lenient parsing)");

        return parse_xml_function_block(function_content);
    }

    return std::nullopt;
}

std::optional<ToolCall> parse_xml_function_block(const std::string& xml_content) {
    // Extract function name and parameters from XML content
    // Input: <function=name><parameter=key>value</parameter>...</function>
    // or: <tool_call><function=name>...</function></tool_call>

    // Extract function name: <function=name>
    std::regex function_regex("<function=([^>]+)>");
    std::smatch function_match;
    if (!std::regex_search(xml_content, function_match, function_regex)) {
        LOG_DEBUG("Failed to extract function name from XML function block");
        return std::nullopt;
    }

    std::string tool_name = function_match[1].str();
    LOG_DEBUG("Extracted XML tool call: " + tool_name);

    // Extract parameters: <parameter=key>value</parameter>
    std::map<std::string, std::any> tool_params;
    std::regex param_regex("<parameter=([^>]+)>([\\s\\S]*?)</parameter>");
    std::sregex_iterator param_it(xml_content.begin(), xml_content.end(), param_regex);
    std::sregex_iterator param_end;

    for (; param_it != param_end; ++param_it) {
        std::string param_name = (*param_it)[1].str();
        std::string param_value = (*param_it)[2].str();

        // Trim leading/trailing whitespace
        size_t start = param_value.find_first_not_of(" \t\n\r");
        size_t end = param_value.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            param_value = param_value.substr(start, end - start + 1);
        }

        tool_params[param_name] = param_value;
        LOG_DEBUG("  Parameter: " + param_name + " = " + param_value.substr(0, 50) + (param_value.length() > 50 ? "..." : ""));
    }

    LOG_DEBUG("Successfully parsed XML function block with " + std::to_string(tool_params.size()) + " parameters");
    return ToolCall(tool_name, tool_params, xml_content, "");
}

std::optional<ToolCall> parse_tool_call(const std::string& response,
                                        const std::vector<std::string>& tool_call_markers) {
    // Try XML format first (Qwen/Unsloth models)
    auto xml_result = parse_xml_tool_call(response);
    if (xml_result.has_value()) {
        return xml_result;
    }

    // Fall back to JSON format
    // First, check if response contains a tool call
    if (!has_tool_call(response, tool_call_markers)) {
        return std::nullopt;
    }

    LOG_DEBUG("Found potential JSON tool call in response");

    try {
        // Extract the JSON part
        std::string json_str = extract_json(response);
        if (json_str.empty()) {
            LOG_DEBUG("Failed to extract JSON from response");
            return std::nullopt;
        }

        // Parse the JSON
        auto json_response = nlohmann::json::parse(json_str);

        // Check for required fields (accept both "parameters" and "arguments")
        if (!json_response.contains("name")) {
            LOG_DEBUG("JSON missing required 'name' field");
            return std::nullopt;
        }

        if (!json_response.contains("parameters") && !json_response.contains("arguments")) {
            LOG_DEBUG("JSON missing required 'parameters' or 'arguments' field");
            return std::nullopt;
        }

        std::string tool_name = json_response["name"];
        // Accept both "parameters" (internal format) and "arguments" (OpenAI format)
        auto params_json = json_response.contains("parameters") ?
                          json_response["parameters"] :
                          json_response["arguments"];

        // Extract optional tool_call_id (used by API backends)
        std::string tool_call_id;
        if (json_response.contains("tool_call_id")) {
            tool_call_id = json_response["tool_call_id"];
        } else if (json_response.contains("id")) {
            tool_call_id = json_response["id"];
        }

        // Convert JSON parameters to std::map<std::string, std::any>
        std::map<std::string, std::any> tool_params;
        for (auto& [key, value] : params_json.items()) {
            if (value.is_string()) {
                tool_params[key] = value.get<std::string>();
            } else if (value.is_number_integer()) {
                tool_params[key] = value.get<int>();
            } else if (value.is_number_float()) {
                tool_params[key] = value.get<double>();
            } else if (value.is_boolean()) {
                tool_params[key] = value.get<bool>();
            } else {
                // For complex types, store as JSON string
                tool_params[key] = value.dump();
            }
        }

        LOG_DEBUG("Successfully parsed JSON tool call: " + tool_name);
        return ToolCall(tool_name, tool_params, json_str, tool_call_id);

    } catch (const nlohmann::json::exception& e) {
        LOG_DEBUG("Failed to parse tool call JSON: " + std::string(e.what()));
        return std::nullopt;
    }
}

} // namespace ToolParser
