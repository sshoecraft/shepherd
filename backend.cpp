
#include "shepherd.h"
#include "backend.h"
#include "backends/channel_parser.h"
#include "backends/chat_template.h"
#include "tools/tool_parser.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Backend::Backend(size_t ctx_size, Session& session, EventCallback cb)
    : context_size(ctx_size), callback(cb) {
    if (!callback) {
        throw std::invalid_argument("Internal error: callback not passed to backend constructor");
    }
}

// Destructor defined here where ChannelParser is complete (for unique_ptr)
Backend::~Backend() = default;

// Initialize marker vectors from virtual methods (called once on first use)
void Backend::ensure_markers_initialized() {
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

// Reset filter state between requests
void Backend::reset_output_state() {
    filter_state = FILTER_NORMAL;
    in_tool_call = false;
    in_thinking = false;
    in_code_block = false;
    skip_to_newline = false;
    json_brace_depth = 0;
    tag_buffer.clear();
    current_tag.clear();
    backtick_buffer.clear();
    buffered_tool_call.clear();
    buffered_thinking.clear();
    output_buffer.clear();
    pending_tool_calls.clear();

    // Initialize channel parser for GPT-OSS harmony format
    // This enables unified channel parsing across CLI, TUI, and API modes
    // Disabled when raw_output is set (to match vLLM behavior)
    const auto* caps = get_chat_template_caps();
    channel_parsing_enabled_ = caps && caps->has_channels && !config->raw_output;

    if (channel_parsing_enabled_) {
        ChannelParsing::ChannelParser::Config cfg;
        cfg.has_channels = true;
        cfg.include_reasoning = show_thinking;
        cfg.channel_start = "<|channel|>";
        cfg.message_start = "<|message|>";
        cfg.channel_end = "<|end|>";
        cfg.turn_start = "<|start|>";

        channel_parser_ = std::make_unique<ChannelParsing::ChannelParser>(cfg);
    } else {
        channel_parser_.reset();
    }
}

// Flush output buffer to callback
// Returns true if should continue, false if cancelled
bool Backend::flush_output_buffer() {
    if (!output_buffer.empty()) {
        // Use CODEBLOCK event when inside code block
        CallbackEvent event = in_code_block ? CallbackEvent::CODEBLOCK : CallbackEvent::CONTENT;
        bool cont = callback(event, output_buffer, "", "");
        output_buffer.clear();
        return cont;
    }
    return true;
}

// Flush any pending output at end of response
void Backend::flush_output() {
    // Flush channel parser first if active
    if (channel_parsing_enabled_ && channel_parser_) {
        channel_parser_->flush([this](ChannelParsing::EventType evt,
                                       const std::string& content,
                                       const std::string& tool_name,
                                       const std::string& args) -> bool {
            switch (evt) {
                case ChannelParsing::EventType::CONTENT:
                case ChannelParsing::EventType::PREAMBLE:
                    // Route through existing output filter
                    return output(content.c_str(), content.length());
                case ChannelParsing::EventType::THINKING:
                    if (show_thinking && callback) {
                        return callback(CallbackEvent::THINKING, content, "", "");
                    }
                    return true;
                case ChannelParsing::EventType::TOOL_CALL:
                    if (callback) {
                        return callback(CallbackEvent::TOOL_CALL, args, tool_name, "");
                    }
                    return true;
            }
            return true;
        });
    }

    // Flush any buffered regular output
    flush_output_buffer();

    // If we're still in a tool call, emit it (incomplete but better than losing it)
    if (in_tool_call && !buffered_tool_call.empty()) {
        emit_tool_call();
    }

    // If we have buffered tag data that never matched, output it
    if (!tag_buffer.empty()) {
        if (config->streaming) {
            callback(CallbackEvent::CONTENT, tag_buffer, "", "");
        }
        tag_buffer.clear();
    }

    // Reset state
    filter_state = FILTER_NORMAL;
    in_tool_call = false;
    in_thinking = false;
}

// Check if buffer exactly matches any marker
bool Backend::matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched) const {
    for (const auto& marker : markers) {
        if (buffer == marker) {
            if (matched) *matched = marker;
            return true;
        }
    }
    return false;
}

// Check if buffer could potentially match a marker (prefix match)
bool Backend::could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const {
    for (const auto& marker : markers) {
        if (marker.length() >= buffer.length() &&
            marker.substr(0, buffer.length()) == buffer) {
            return true;
        }
    }
    return false;
}

// Parse buffered tool call and emit TOOL_REQUEST callback
void Backend::emit_tool_call() {
    if (!callback || buffered_tool_call.empty()) {
        buffered_tool_call.clear();
        return;
    }

    // Try to parse the tool call
    std::string tool_name;
    std::string tool_args;
    std::string tool_id;

    // Check if it's JSON format (starts with {)
    size_t json_start = buffered_tool_call.find('{');
    if (json_start != std::string::npos) {
        std::string json_str = buffered_tool_call.substr(json_start);
        // Find the end of JSON (balanced braces)
        int depth = 0;
        size_t json_end = 0;
        for (size_t i = 0; i < json_str.length(); i++) {
            if (json_str[i] == '{') depth++;
            if (json_str[i] == '}') depth--;
            if (depth == 0) {
                json_end = i + 1;
                break;
            }
        }
        if (json_end > 0) {
            json_str = json_str.substr(0, json_end);
            try {
                json j = json::parse(json_str);
                if (j.contains("name")) {
                    tool_name = j["name"].get<std::string>();
                }
                if (j.contains("arguments")) {
                    if (j["arguments"].is_string()) {
                        tool_args = j["arguments"].get<std::string>();
                    } else {
                        tool_args = j["arguments"].dump();
                    }
                } else if (j.contains("parameters")) {
                    if (j["parameters"].is_string()) {
                        tool_args = j["parameters"].get<std::string>();
                    } else {
                        tool_args = j["parameters"].dump();
                    }
                }
                if (j.contains("id")) {
                    tool_id = j["id"].get<std::string>();
                }
            } catch (...) {
                // JSON parse failed, use raw content
                tool_args = json_str;
            }
        }
    }

    // Check for XML format: <function=name>args</function> (Qwen3-Coder style)
    if (tool_name.empty()) {
        size_t func_start = buffered_tool_call.find("<function=");
        if (func_start != std::string::npos) {
            size_t name_start = func_start + 10;  // length of "<function="
            size_t name_end = buffered_tool_call.find('>', name_start);
            if (name_end != std::string::npos) {
                tool_name = buffered_tool_call.substr(name_start, name_end - name_start);
                // Extract arguments between > and </function>
                size_t args_start = name_end + 1;
                size_t args_end = buffered_tool_call.find("</function>", args_start);
                if (args_end != std::string::npos) {
                    tool_args = buffered_tool_call.substr(args_start, args_end - args_start);
                    // Trim whitespace from args
                    size_t first = tool_args.find_first_not_of(" \t\n\r");
                    size_t last = tool_args.find_last_not_of(" \t\n\r");
                    if (first != std::string::npos && last != std::string::npos) {
                        tool_args = tool_args.substr(first, last - first + 1);
                    } else {
                        tool_args = "{}";  // Empty args
                    }
                } else {
                    tool_args = "{}";  // No closing tag, assume empty args
                }
                dout(2) << "Parsed XML tool call: name=" << tool_name << ", args=" << tool_args << std::endl;
            }
        }
    }

    // Generate tool_id if not provided
    if (tool_id.empty()) {
        static int tool_call_counter = 0;
        tool_id = "call_" + std::to_string(++tool_call_counter);
    }

    // Check if tool name is valid - if not, output as content instead
    if (tool_name.empty() || (!valid_tool_names.empty() && valid_tool_names.find(tool_name) == valid_tool_names.end())) {
        // Not a valid tool - output the raw content
        dout(2) << "emit_tool_call: tool '" << tool_name << "' not in valid_tool_names, outputting as content" << std::endl;
        callback(CallbackEvent::CONTENT, buffered_tool_call, "", "");
        buffered_tool_call.clear();
        return;
    }

    // Store parsed tool call for later use in assistant message's tool_calls_json
    ToolParser::ToolCall tc;
    tc.name = tool_name;
    tc.tool_call_id = tool_id;
    tc.raw_json = tool_args;
    pending_tool_calls.push_back(tc);

    // Emit the tool request
    callback(CallbackEvent::TOOL_CALL, tool_args, tool_name, tool_id);

    buffered_tool_call.clear();
}

// Process output through channel parser (for GPT-OSS harmony format)
// If model has channels: routes through ChannelParser first, then output()
// If no channels: falls back to output() directly
// Returns true to continue, false if cancelled or stop requested
bool Backend::process_output(const char* text, size_t len) {
    // In server mode, skip all filtering - client expects raw markdown
    if (g_server_mode) {
        return callback(CallbackEvent::CONTENT, std::string(text, len), "", "");
    }

    if (!channel_parsing_enabled_ || !channel_parser_) {
        // No channel parsing - use existing output filter directly
        return output(text, len);
    }

    // Route through channel parser first
    std::string delta(text, len);
    bool continue_generation = channel_parser_->process(delta,
        [this](ChannelParsing::EventType evt, const std::string& content,
               const std::string& tool_name, const std::string& args) -> bool {

            switch (evt) {
                case ChannelParsing::EventType::CONTENT:
                case ChannelParsing::EventType::PREAMBLE:
                    // Pass final channel content through existing output filter
                    // (handles any nested tool calls or thinking markers)
                    return output(content.c_str(), content.length());

                case ChannelParsing::EventType::THINKING:
                    // Analysis channel - emit as thinking if enabled
                    if (show_thinking && callback) {
                        return callback(CallbackEvent::THINKING, content, "", "");
                    }
                    return true;  // Suppress if not showing thinking

                case ChannelParsing::EventType::TOOL_CALL: {
                    // Commentary channel tool call - already extracted
                    // Generate tool call ID and store for later use
                    static int channel_tool_call_counter = 0;
                    std::string tool_id = "call_" + std::to_string(++channel_tool_call_counter);

                    // Store in pending_tool_calls for assistant message
                    ToolParser::ToolCall tc;
                    tc.name = tool_name;
                    tc.tool_call_id = tool_id;
                    tc.raw_json = args;
                    pending_tool_calls.push_back(tc);

                    if (callback) {
                        return callback(CallbackEvent::TOOL_CALL, args, tool_name, tool_id);
                    }
                    return true;
                }
            }
            return true;
        });

    return continue_generation;
}

// Unified output function - filters tool calls and thinking blocks
// Returns true to continue, false if callback requested cancellation
bool Backend::output(const char* text, size_t len) {
    // Fast path for raw mode - skip all filtering
    if (config->raw_output) {
        if (callback) {
            return callback(CallbackEvent::CONTENT, std::string(text, len), "", "");
        }
        return true;
    }

    ensure_markers_initialized();
    bool cancelled = false;

    dout(2) << "output() called: len=" << len << " filter_state=" << filter_state
            << " text=[" << std::string(text, std::min(len, (size_t)50)) << "]" << std::endl;

    for (size_t i = 0; i < len; i++) {
        char c = text[i];

        // Track backticks for code block detection (``` toggles code block mode)
        // Suppress the ``` markers entirely from output
        if (c == '`') {
            backtick_buffer += c;
            if (backtick_buffer.length() >= 3) {
                in_code_block = !in_code_block;
                backtick_buffer.clear();
                // Skip any language identifier after opening ```
                if (in_code_block) {
                    skip_to_newline = true;
                }
            }
            continue;  // Don't add backticks to output
        } else {
            // If we had partial backticks, flush them as content
            if (!backtick_buffer.empty()) {
                output_buffer += backtick_buffer;
                backtick_buffer.clear();
            }
        }

        // Skip language identifier on code block opening line
        if (skip_to_newline) {
            if (c == '\n') {
                skip_to_newline = false;
            }
            continue;
        }

        switch (filter_state) {
            case FILTER_NORMAL:
                if (c == '<' && !in_code_block) {
                    // Flush any buffered output before starting tag detection
                    if (!flush_output_buffer()) { cancelled = true; break; }
                    tag_buffer = "<";
                    filter_state = FILTER_DETECTING_TAG;
                } else if (c == '{' && !in_code_block) {
                    // JSON tool call - enter tool call mode directly
                    // (single-char marker, no need for FILTER_DETECTING_TAG)
                    if (!flush_output_buffer()) { cancelled = true; break; }
                    in_tool_call = true;
                    current_tag = "{";
                    filter_state = FILTER_IN_TOOL_CALL;
                    buffered_tool_call = "{";
                    json_brace_depth = 1;
                } else {
                    // Buffer regular output for batching
                    output_buffer += c;
                    // Flush on newlines for responsive streaming
                    if (c == '\n') {
                        if (!flush_output_buffer()) { cancelled = true; break; }
                    }
                }
                break;

            case FILTER_DETECTING_TAG: {
                tag_buffer += c;

                // Check for closing tag start
                if (tag_buffer.length() == 2 && tag_buffer == "</") {
                    filter_state = FILTER_CHECKING_CLOSE;
                    break;
                }

                // Check if we've matched a tool call start marker
                std::string matched_marker;
                bool is_tool_call_match = false;

                if (matches_any(tag_buffer, tool_call_start_markers, &matched_marker)) {
                    // For single-char markers like '{', enter tool call mode immediately
                    // (no boundary char to wait for - JSON content follows directly)
                    if (matched_marker.length() == 1 && matched_marker[0] == '{') {
                        is_tool_call_match = true;
                    }
                    // For multi-char markers, wait for boundary char to confirm
                } else {
                    // Check if we're one char past a marker (boundary check)
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

                // Check if we've matched a thinking start marker
                if (matches_any(tag_buffer, thinking_start_markers, &matched_marker)) {
                    in_thinking = true;
                    current_tag = matched_marker;
                    filter_state = FILTER_IN_THINKING;
                    if (show_thinking && callback) {
                        callback(CallbackEvent::THINKING, tag_buffer, "", "");
                    }
                    tag_buffer.clear();
                    break;
                }

                // Check if tag could still match something
                bool could_match = could_match_any(tag_buffer, tool_call_start_markers) ||
                                   could_match_any(tag_buffer, thinking_start_markers);

                if (!could_match) {
                    // Not a special tag, output buffered content
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
                    // For JSON tool calls, track brace depth
                    if (current_tag.length() > 0 && current_tag[0] == '{') {
                        if (c == '{') json_brace_depth++;
                        if (c == '}') json_brace_depth--;
                        if (json_brace_depth == 0) {
                            // JSON complete, emit tool call
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
                    if (show_thinking && callback) {
                        std::string s(1, c);
                        callback(CallbackEvent::THINKING, s, "", "");
                    } else {
                        buffered_thinking += c;
                    }
                }
                break;

            case FILTER_CHECKING_CLOSE: {
                bool could_close = false;  // Declare before any gotos
                tag_buffer += c;

                // Check if closing tag matches expected end marker
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
                            if (show_thinking && callback) {
                                callback(CallbackEvent::THINKING, tag_buffer, "", "");
                            }
                            in_thinking = false;
                            filter_state = FILTER_NORMAL;
                            tag_buffer.clear();
                            goto next_char;
                        }
                    }
                }

                // Check for partial match
                if (in_tool_call) {
                    could_close = could_match_any(tag_buffer, tool_call_end_markers);
                } else if (in_thinking) {
                    could_close = could_match_any(tag_buffer, thinking_end_markers);
                }

                if (!could_close) {
                    // False alarm, not a closing tag
                    if (in_tool_call) {
                        buffered_tool_call += tag_buffer;
                        filter_state = FILTER_IN_TOOL_CALL;
                    } else if (in_thinking) {
                        if (show_thinking && callback) {
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

        // Check for cancellation after switch
        if (cancelled) break;
    }

    // Flush any buffered output at end of each chunk
    if (!cancelled) {
        flush_output_buffer();
    }

    return !cancelled;
}

#if 0
void Backend::update_token_counts_from_api(int prompt_tokens, int completion_tokens, int estimated_prompt_tokens) {
    // Update user message with actual prompt token count if different
    if (prompt_tokens != estimated_prompt_tokens) {
        auto& messages = context_manager_->get_messages();
        if (!messages.empty() && messages.back().role == Message::USER) {
            messages.back().token_count = prompt_tokens;
            // Recalculate total token count since we changed a message
            context_manager_->recalculate_total_tokens();
        }
    }

    // Note: completion tokens are handled when creating the assistant message
    // This method just handles updating the prompt token count
}
#endif
