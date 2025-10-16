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
    // Tool call detection: response must have JSON on its own line
    // This prevents false positives when model explains tool usage
    bool has_json_on_own_line = false;
    std::istringstream stream(response);
    std::string line;

    while (std::getline(stream, line)) {
        // Trim leading whitespace from this line
        size_t start = line.find_first_not_of(" \t\r");
        if (start != std::string::npos) {
            std::string trimmed_line = line.substr(start);

            // Check if this line starts with {
            if (!trimmed_line.empty() && trimmed_line[0] == '{') {
                has_json_on_own_line = true;
                LOG_DEBUG("Found JSON on its own line");
                break;
            }

            // Check for tool markers
            if (!tool_call_markers.empty()) {
                for (const auto& marker : tool_call_markers) {
                    if (trimmed_line.find(marker) == 0) {
                        has_json_on_own_line = true;
                        LOG_DEBUG("Found tool call marker on its own line: " + marker);
                        break;
                    }
                }
                if (has_json_on_own_line) break;
            }
        }
    }

    // Must have JSON on own line AND be able to extract and parse valid JSON
    // with "name" and "parameters" fields
    if (!has_json_on_own_line) {
        return false;
    }

    // Try to extract and parse the JSON to verify it's a valid tool call
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
        if (json_response.contains("name") && json_response.contains("parameters")) {
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

            // Extract function name and parameters from the content
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

        // Check for required fields
        if (!json_response.contains("name") || !json_response.contains("parameters")) {
            LOG_DEBUG("JSON missing required 'name' or 'parameters' fields");
            return std::nullopt;
        }

        std::string tool_name = json_response["name"];
        auto params_json = json_response["parameters"];

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
