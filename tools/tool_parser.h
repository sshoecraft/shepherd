#pragma once

#include <string>
#include <map>
#include <vector>
#include <any>
#include <optional>

namespace ToolParser {

/// @brief Represents a parsed tool call from LLM response
struct ToolCall {
    std::string name;
    std::map<std::string, std::any> parameters;
    std::string raw_json;  // Original JSON string for debugging
    std::string tool_call_id;  // Optional tool call ID (used by API backends like OpenAI)
    std::string content;  // Text content before the tool call (e.g., "Let me look that up for you")

    ToolCall() = default;
    ToolCall(const std::string& n, const std::map<std::string, std::any>& p, const std::string& json = "", const std::string& id = "", const std::string& c = "")
        : name(n), parameters(p), raw_json(json), tool_call_id(id), content(c) {}
};

/// @brief Parse a tool call from LLM response
/// @param response Raw response text from LLM
/// @param tool_call_markers Optional list of markers that indicate tool calls (e.g., "<tool_call>")
/// @return ToolCall if found, std::nullopt otherwise
std::optional<ToolCall> parse_tool_call(const std::string& response,
                                        const std::vector<std::string>& tool_call_markers = {});

/// @brief Detect if response contains a tool call
/// @param response Raw response text from LLM
/// @param tool_call_markers Optional list of markers that indicate tool calls
/// @return True if tool call detected
bool has_tool_call(const std::string& response,
                  const std::vector<std::string>& tool_call_markers = {});

/// @brief Extract JSON string from response by matching braces
/// @param response Response containing JSON
/// @return Extracted JSON string, or empty string if not found
std::string extract_json(const std::string& response);

} // namespace ToolParser
