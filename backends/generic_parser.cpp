// GenericParser implementation
// Ported from GpuBackend::output() state machine

#include "generic_parser.h"
#include "harmony_parser.h"
#include "../tools/tool_parser.h"
#include <nlohmann/json.hpp>

namespace StreamParser {

GenericParser::GenericParser(
    const std::vector<std::string>& tool_start,
    const std::vector<std::string>& tool_end,
    const std::vector<std::string>& thinking_start,
    const std::vector<std::string>& thinking_end)
    : tool_start_markers(tool_start)
    , tool_end_markers(tool_end)
    , thinking_start_markers(thinking_start)
    , thinking_end_markers(thinking_end) {

    // Add common fallback markers if none provided
    if (tool_start_markers.empty()) {
        tool_start_markers = {"<tool_call>", "<function_call>", "<tool_use>", "{"};
    }
    if (tool_end_markers.empty()) {
        tool_end_markers = {"</tool_call>", "</function_call>", "</tool_use>"};
    }
}

void GenericParser::reset() {
    state = State::NORMAL;
    in_tool_call = false;
    in_thinking = false;
    json_brace_depth = 0;

    tag_buffer.clear();
    current_tag.clear();
    buffered_tool_call.clear();
    buffered_thinking.clear();

    pending_content.clear();
    pending_reasoning.clear();
    pending_tool_calls.clear();
}

bool GenericParser::process(const std::string& token) {
    for (char c : token) {
        process_char(c);
    }
    // GenericParser never signals stop - generation continues until EOG token
    return false;
}

void GenericParser::process_char(char c) {
    switch (state) {
        case State::NORMAL:
            if (c == '<') {
                tag_buffer = "<";
                state = State::DETECTING_TAG;
            } else if (c == '{') {
                // Check if { is a tool call start marker
                bool is_tool_marker = false;
                for (const auto& marker : tool_start_markers) {
                    if (marker == "{") {
                        is_tool_marker = true;
                        break;
                    }
                }
                if (is_tool_marker) {
                    in_tool_call = true;
                    current_tag = "{";
                    state = State::IN_TOOL_CALL;
                    buffered_tool_call = "{";
                    json_brace_depth = 1;
                } else {
                    pending_content += c;
                }
            } else {
                pending_content += c;
            }
            break;

        case State::DETECTING_TAG: {
            tag_buffer += c;

            // Check for closing tag start
            if (tag_buffer.length() == 2 && tag_buffer == "</") {
                state = State::CHECKING_CLOSE;
                break;
            }

            std::string matched_marker;
            bool is_tool_call_match = false;

            // Check for exact marker match
            if (matches_any(tag_buffer, tool_start_markers, &matched_marker)) {
                if (matched_marker.length() == 1 && matched_marker[0] == '{') {
                    is_tool_call_match = true;
                }
            } else {
                // Check for marker + boundary character (e.g., "<tool_call>")
                for (const auto& marker : tool_start_markers) {
                    if (tag_buffer.length() == marker.length() + 1 &&
                        tag_buffer.substr(0, marker.length()) == marker) {
                        char boundary = tag_buffer.back();
                        if (boundary == '>' || boundary == ' ' || boundary == '/' ||
                            boundary == '\n' || boundary == '\t' || boundary == '\r' ||
                            boundary == ':') {
                            matched_marker = marker;
                            is_tool_call_match = true;
                            break;
                        }
                    }
                }
            }

            if (is_tool_call_match) {
                in_tool_call = true;
                current_tag = matched_marker;
                state = State::IN_TOOL_CALL;
                buffered_tool_call = tag_buffer;
                if (!matched_marker.empty() && matched_marker[0] == '{') {
                    json_brace_depth = 1;
                }
                tag_buffer.clear();
                break;
            }

            // Check for thinking marker match
            if (matches_any(tag_buffer, thinking_start_markers, &matched_marker)) {
                in_thinking = true;
                current_tag = matched_marker;
                state = State::IN_THINKING;
                // Don't include opening tag in reasoning output
                tag_buffer.clear();
                break;
            }

            // Check if buffer could still match a marker
            bool could_match = could_match_any(tag_buffer, tool_start_markers) ||
                               could_match_any(tag_buffer, thinking_start_markers);

            if (!could_match) {
                // No match possible - output buffered content
                pending_content += tag_buffer;
                tag_buffer.clear();
                state = State::NORMAL;
            }
            break;
        }

        case State::IN_TOOL_CALL:
            if (c == '<') {
                tag_buffer = "<";
                state = State::CHECKING_CLOSE;
            } else {
                buffered_tool_call += c;
                if (!current_tag.empty() && current_tag[0] == '{') {
                    if (c == '{') json_brace_depth++;
                    if (c == '}') json_brace_depth--;
                    if (json_brace_depth == 0) {
                        emit_tool_call();
                        in_tool_call = false;
                        state = State::NORMAL;
                    }
                }
            }
            break;

        case State::IN_THINKING:
            if (c == '<') {
                tag_buffer = "<";
                state = State::CHECKING_CLOSE;
            } else {
                pending_reasoning += c;
            }
            break;

        case State::CHECKING_CLOSE: {
            tag_buffer += c;

            // Check for tool call end marker
            if (in_tool_call) {
                for (const auto& end_marker : tool_end_markers) {
                    if (tag_buffer == end_marker) {
                        buffered_tool_call += tag_buffer;
                        emit_tool_call();
                        in_tool_call = false;
                        state = State::NORMAL;
                        tag_buffer.clear();
                        return;
                    }
                }
            }

            // Check for thinking end marker
            if (in_thinking) {
                for (const auto& end_marker : thinking_end_markers) {
                    if (tag_buffer == end_marker) {
                        // Don't include closing tag in reasoning output
                        in_thinking = false;
                        state = State::NORMAL;
                        tag_buffer.clear();
                        return;
                    }
                }
            }

            // Check if buffer could still match a closing marker
            bool could_close = false;
            if (in_tool_call) {
                could_close = could_match_any(tag_buffer, tool_end_markers);
            } else if (in_thinking) {
                could_close = could_match_any(tag_buffer, thinking_end_markers);
            }

            if (!could_close) {
                // Not a closing tag - add buffer to appropriate accumulator
                if (in_tool_call) {
                    buffered_tool_call += tag_buffer;
                    state = State::IN_TOOL_CALL;
                } else if (in_thinking) {
                    pending_reasoning += tag_buffer;
                    state = State::IN_THINKING;
                }
                tag_buffer.clear();
            }
            break;
        }
    }
}

bool GenericParser::matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched) const {
    for (const auto& marker : markers) {
        if (buffer == marker) {
            if (matched) *matched = marker;
            return true;
        }
    }
    return false;
}

bool GenericParser::could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const {
    for (const auto& marker : markers) {
        if (marker.length() >= buffer.length() &&
            marker.substr(0, buffer.length()) == buffer) {
            return true;
        }
    }
    return false;
}

void GenericParser::emit_tool_call() {
    if (buffered_tool_call.empty()) {
        return;
    }

    // Use ToolParser for complete parsing of all formats
    auto parsed = ToolParser::parse_tool_call(buffered_tool_call, tool_start_markers);

    if (!parsed) {
        // Failed to parse - output as content
        pending_content += buffered_tool_call;
        buffered_tool_call.clear();
        return;
    }

    std::string tool_id = parsed->tool_call_id;
    if (tool_id.empty()) {
        static int tool_call_counter = 0;
        tool_id = "call_" + std::to_string(++tool_call_counter);
    }

    // Convert parameters to JSON string
    std::string params_json;
    if (!parsed->raw_json.empty()) {
        params_json = parsed->raw_json;
    } else if (!parsed->parameters.empty()) {
        nlohmann::json j;
        for (const auto& [key, value] : parsed->parameters) {
            if (value.type() == typeid(std::string)) {
                j[key] = std::any_cast<std::string>(value);
            } else if (value.type() == typeid(int)) {
                j[key] = std::any_cast<int>(value);
            } else if (value.type() == typeid(double)) {
                j[key] = std::any_cast<double>(value);
            } else if (value.type() == typeid(bool)) {
                j[key] = std::any_cast<bool>(value);
            }
        }
        params_json = j.dump();
    } else {
        params_json = "{}";
    }

    pending_tool_calls.push_back({parsed->name, params_json, tool_id});
    buffered_tool_call.clear();
}

std::string GenericParser::get_content_delta() {
    std::string delta = std::move(pending_content);
    pending_content.clear();
    return delta;
}

std::string GenericParser::get_reasoning_delta() {
    std::string delta = std::move(pending_reasoning);
    pending_reasoning.clear();
    return delta;
}

std::vector<ToolCall> GenericParser::get_tool_calls() {
    std::vector<ToolCall> calls = std::move(pending_tool_calls);
    pending_tool_calls.clear();
    return calls;
}

bool GenericParser::has_pending_content() const {
    return !tag_buffer.empty() || !buffered_tool_call.empty() || !buffered_thinking.empty();
}

void GenericParser::flush() {
    // Flush any incomplete tool call
    if (in_tool_call && !buffered_tool_call.empty()) {
        emit_tool_call();
    }

    // Flush tag buffer as content
    if (!tag_buffer.empty()) {
        pending_content += tag_buffer;
        tag_buffer.clear();
    }

    state = State::NORMAL;
    in_tool_call = false;
    in_thinking = false;
}

// Factory function to create appropriate parser
std::unique_ptr<Parser> create_parser(
    bool has_harmony_channels,
    const std::vector<std::string>& tool_start_markers,
    const std::vector<std::string>& tool_end_markers,
    const std::vector<std::string>& thinking_start_markers,
    const std::vector<std::string>& thinking_end_markers) {

    if (has_harmony_channels) {
        // Include harmony_parser.h here to avoid circular dependency in header
        return std::make_unique<HarmonyParser>();
    } else {
        return std::make_unique<GenericParser>(
            tool_start_markers,
            tool_end_markers,
            thinking_start_markers,
            thinking_end_markers
        );
    }
}

} // namespace StreamParser
