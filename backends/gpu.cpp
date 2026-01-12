
#include "gpu.h"
#include "generic_parser.h"
#include "chat_template.h"
#include "../shepherd.h"
#include "../tools/tool_parser.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Use config->thinking instead of config->thinking

// Returns the length of valid UTF-8 in text
// If validate_utf8(text) < text.size(), there's an incomplete multi-byte char at the end
static size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx - needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx - needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx - needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    return len;
}

GpuBackend::GpuBackend(size_t ctx_size, Session& session, EventCallback cb)
    : Backend(ctx_size, session, cb) {
    is_gpu = true;
}

GpuBackend::~GpuBackend() = default;

void GpuBackend::ensure_markers_initialized() {
    if (markers_initialized) return;

    tool_call_start_markers = get_tool_call_markers();
    tool_call_end_markers = get_tool_call_end_markers();
    thinking_start_markers = get_thinking_start_markers();
    thinking_end_markers = get_thinking_end_markers();

    // Add common fallback markers if backend doesn't provide any
    if (tool_call_start_markers.empty()) {
        tool_call_start_markers = {"<tool_call>", "<function_call>", "<tool_use>", "{"};
    }
    if (tool_call_end_markers.empty()) {
        tool_call_end_markers = {"</tool_call>", "</function_call>", "</tool_use>"};
    }

    markers_initialized = true;
}

void GpuBackend::reset_output_state() {
    // Reset common filter state (backticks, output_buffer, in_code_block, etc.)
    Backend::reset_output_state();

    // Reset GPU-specific state
    filter_state = FILTER_NORMAL;
    in_tool_call = false;
    in_thinking = false;
    json_brace_depth = 0;
    tag_buffer.clear();
    current_tag.clear();
    buffered_tool_call.clear();
    buffered_thinking.clear();
    utf8_buffer.clear();

    // Check if channel parsing is enabled (harmony parsing now done in LlamaCppBackend)
    const auto* caps = get_chat_template_caps();
    channelParsingEnabled = caps && caps->has_channels && !config->raw_output;

    // Create streaming parser based on model capabilities
    // Harmony models use HarmonyParser (O(n) channel-based)
    // Non-harmony models use GenericParser (tool calls, thinking tags)
    ensure_markers_initialized();
    parser = StreamParser::create_parser(
        channelParsingEnabled,
        tool_call_start_markers,
        tool_call_end_markers,
        thinking_start_markers,
        thinking_end_markers
    );
}

void GpuBackend::flush_output() {
    // For harmony models, flushing is handled by LlamaCppBackend
    // This handles non-harmony path: flush tag detection buffers
    // Flush common filter buffer (backticks, output_buffer)
    Backend::flush_output();

    // Flush the streaming parser and emit any remaining content
    if (parser) {
        parser->flush();

        // Emit any remaining content delta after flush
        std::string content_delta = parser->get_content_delta();
        if (!content_delta.empty()) {
            filter(content_delta.c_str(), content_delta.length());
        }

        // Emit any remaining reasoning delta
        std::string reasoning_delta = parser->get_reasoning_delta();
        if (!reasoning_delta.empty() && config->thinking ) {
            callback(CallbackEvent::THINKING, reasoning_delta, "", "");
        }

        // Emit any remaining tool calls
        auto tool_calls = parser->get_tool_calls();
        for (const auto& tc : tool_calls) {
            record_tool_call(tc.name, tc.arguments, tc.id);
        }
    }

    if (in_tool_call && !buffered_tool_call.empty()) {
        emit_tool_call();
    }

    if (!tag_buffer.empty()) {
        if (config->streaming) {
            callback(CallbackEvent::CONTENT, tag_buffer, "", "");
        }
        tag_buffer.clear();
    }

    filter_state = FILTER_NORMAL;
    in_tool_call = false;
    in_thinking = false;
}

bool GpuBackend::matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched) const {
    for (const auto& marker : markers) {
        if (buffer == marker) {
            if (matched) *matched = marker;
            return true;
        }
    }
    return false;
}

bool GpuBackend::could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const {
    for (const auto& marker : markers) {
        if (marker.length() >= buffer.length() &&
            marker.substr(0, buffer.length()) == buffer) {
            return true;
        }
    }
    return false;
}

void GpuBackend::emit_tool_call() {
    if (!callback || buffered_tool_call.empty()) {
        buffered_tool_call.clear();
        return;
    }

    // Use ToolParser for complete parsing of all formats (JSON, XML, bracket, etc.)
    auto parsed = ToolParser::parse_tool_call(buffered_tool_call, tool_call_start_markers);

    if (!parsed) {
        dout(2) << "emit_tool_call: failed to parse tool call, outputting as content" << std::endl;
        callback(CallbackEvent::CONTENT, buffered_tool_call, "", "");
        buffered_tool_call.clear();
        return;
    }

    std::string tool_name = parsed->name;
    std::string tool_id = parsed->tool_call_id;

    // Generate tool ID if not provided
    if (tool_id.empty()) {
        static int tool_call_counter = 0;
        tool_id = "call_" + std::to_string(++tool_call_counter);
    }

    // Check if tool name is valid
    if (tool_name.empty() || (!valid_tool_names.empty() && valid_tool_names.find(tool_name) == valid_tool_names.end())) {
        dout(2) << "emit_tool_call: tool '" << tool_name << "' not in valid_tool_names, outputting as content" << std::endl;
        callback(CallbackEvent::CONTENT, buffered_tool_call, "", "");
        buffered_tool_call.clear();
        return;
    }

    // Convert parsed parameters to JSON string
    std::string params_json;
    if (!parsed->raw_json.empty()) {
        params_json = parsed->raw_json;
    } else if (!parsed->parameters.empty()) {
        // Build JSON from parameters map
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

    dout(2) << "emit_tool_call: name=" << tool_name << ", params=" << params_json << std::endl;

    // Record for emission after STOP (don't emit here)
    record_tool_call(tool_name, params_json, tool_id);
    buffered_tool_call.clear();
}

bool GpuBackend::output(const char* text, size_t len) {
    // Handle incomplete UTF-8 sequences from previous call
    std::string combined;
    if (!utf8_buffer.empty()) {
        combined = utf8_buffer + std::string(text, len);
        utf8_buffer.clear();
        text = combined.c_str();
        len = combined.length();
    }

    // Check for incomplete UTF-8 at the end and buffer it
    size_t valid_len = validate_utf8(std::string(text, len));
    if (valid_len < len) {
        // Buffer the incomplete UTF-8 bytes for next call
        utf8_buffer = std::string(text + valid_len, len - valid_len);
        len = valid_len;
        if (len == 0) return true;  // Nothing complete to output yet
    }

    // Fast path for raw mode - bypass all filtering
    if (config->raw_output) {
        return callback(CallbackEvent::CONTENT, std::string(text, len), "", "");
    }

    // For harmony models, parsing is now handled in LlamaCppBackend
    // This handles non-harmony path - tag filtering for tool calls and thinking blocks
    // Then calls filter() for backtick handling and buffering
    ensure_markers_initialized();
    bool cancelled = false;

    dout(2) << "output() called (non-harmony): len=" << len << " filter_state=" << filter_state
            << " text=[" << std::string(text, std::min(len, (size_t)50)) << "]" << std::endl;

    for (size_t i = 0; i < len; i++) {
        char c = text[i];

        // Track backticks for code block detection
        if (c == '`') {
            backtick_buffer += c;
            if (backtick_buffer.length() >= 3) {
                in_code_block = !in_code_block;
                backtick_buffer.clear();
                if (in_code_block) {
                    skip_to_newline = true;
                }
            }
            continue;
        } else {
            if (!backtick_buffer.empty()) {
                output_buffer += backtick_buffer;
                backtick_buffer.clear();
            }
        }

        if (skip_to_newline) {
            if (c == '\n') {
                skip_to_newline = false;
            }
            continue;
        }

        switch (filter_state) {
            case FILTER_NORMAL:
                if (c == '<' && !in_code_block) {
                    if (!flush_filter_buffer()) { cancelled = true; break; }
                    tag_buffer = "<";
                    filter_state = FILTER_DETECTING_TAG;
                } else if (c == '{' && !in_code_block) {
                    if (!flush_filter_buffer()) { cancelled = true; break; }
                    in_tool_call = true;
                    current_tag = "{";
                    filter_state = FILTER_IN_TOOL_CALL;
                    buffered_tool_call = "{";
                    json_brace_depth = 1;
                } else {
                    output_buffer += c;
                    if (c == '\n') {
                        if (!flush_filter_buffer()) { cancelled = true; break; }
                    }
                }
                break;

            case FILTER_DETECTING_TAG: {
                tag_buffer += c;

                if (tag_buffer.length() == 2 && tag_buffer == "</") {
                    filter_state = FILTER_CHECKING_CLOSE;
                    break;
                }

                std::string matched_marker;
                bool is_tool_call_match = false;

                if (matches_any(tag_buffer, tool_call_start_markers, &matched_marker)) {
                    if (matched_marker.length() == 1 && matched_marker[0] == '{') {
                        is_tool_call_match = true;
                    }
                } else {
                    for (const auto& marker : tool_call_start_markers) {
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
                    filter_state = FILTER_IN_TOOL_CALL;
                    buffered_tool_call = tag_buffer;
                    if (matched_marker.length() > 0 && matched_marker[0] == '{') {
                        json_brace_depth = 1;
                    }
                    tag_buffer.clear();
                    break;
                }

                if (matches_any(tag_buffer, thinking_start_markers, &matched_marker)) {
                    in_thinking = true;
                    current_tag = matched_marker;
                    filter_state = FILTER_IN_THINKING;
                    if (config->thinking ) {
                        callback(CallbackEvent::THINKING, tag_buffer, "", "");
                    }
                    tag_buffer.clear();
                    break;
                }

                bool could_match = could_match_any(tag_buffer, tool_call_start_markers) ||
                                   could_match_any(tag_buffer, thinking_start_markers);

                if (!could_match) {
                    if (!callback(CallbackEvent::CONTENT, tag_buffer, "", "")) {
                        cancelled = true;
                    }
                    tag_buffer.clear();
                    filter_state = FILTER_NORMAL;
                }
                break;
            }

            case FILTER_IN_TOOL_CALL:
                if (c == '<') {
                    tag_buffer = "<";
                    filter_state = FILTER_CHECKING_CLOSE;
                } else {
                    buffered_tool_call += c;
                    if (current_tag.length() > 0 && current_tag[0] == '{') {
                        if (c == '{') json_brace_depth++;
                        if (c == '}') json_brace_depth--;
                        if (json_brace_depth == 0) {
                            emit_tool_call();
                            in_tool_call = false;
                            filter_state = FILTER_NORMAL;
                        }
                    }
                }
                break;

            case FILTER_IN_THINKING:
                if (c == '<') {
                    tag_buffer = "<";
                    filter_state = FILTER_CHECKING_CLOSE;
                } else {
                    if (config->thinking ) {
                        std::string s(1, c);
                        callback(CallbackEvent::THINKING, s, "", "");
                    } else {
                        buffered_thinking += c;
                    }
                }
                break;

            case FILTER_CHECKING_CLOSE: {
                bool could_close = false;
                tag_buffer += c;

                if (in_tool_call) {
                    for (const auto& end_marker : tool_call_end_markers) {
                        if (tag_buffer == end_marker) {
                            buffered_tool_call += tag_buffer;
                            emit_tool_call();
                            in_tool_call = false;
                            filter_state = FILTER_NORMAL;
                            tag_buffer.clear();
                            goto next_char;
                        }
                    }
                }

                if (in_thinking) {
                    for (const auto& end_marker : thinking_end_markers) {
                        if (tag_buffer == end_marker) {
                            if (config->thinking ) {
                                callback(CallbackEvent::THINKING, tag_buffer, "", "");
                            }
                            in_thinking = false;
                            filter_state = FILTER_NORMAL;
                            tag_buffer.clear();
                            goto next_char;
                        }
                    }
                }

                if (in_tool_call) {
                    could_close = could_match_any(tag_buffer, tool_call_end_markers);
                } else if (in_thinking) {
                    could_close = could_match_any(tag_buffer, thinking_end_markers);
                }

                if (!could_close) {
                    if (in_tool_call) {
                        buffered_tool_call += tag_buffer;
                        filter_state = FILTER_IN_TOOL_CALL;
                    } else if (in_thinking) {
                        if (config->thinking ) {
                            callback(CallbackEvent::THINKING, tag_buffer, "", "");
                        } else {
                            buffered_thinking += tag_buffer;
                        }
                        filter_state = FILTER_IN_THINKING;
                    }
                    tag_buffer.clear();
                }
                next_char:
                break;
            }
        }

        if (cancelled) break;
    }

    if (!cancelled) {
        flush_filter_buffer();
    }

    return !cancelled;
}
