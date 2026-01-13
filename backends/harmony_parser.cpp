// HarmonyParser implementation
// O(n) character-by-character state machine for GPT-OSS harmony format

#include "harmony_parser.h"
#include "../tools/tool_parser.h"
#include "../shepherd.h"  // for dout
#include <algorithm>
#include <regex>

namespace StreamParser {

void HarmonyParser::reset() {
    parse_state = ParseState::EXPECT_START;
    current_channel = Channel::NONE;
    tool_recipient.clear();
    constraint_type.clear();

    char_state = CharState::NORMAL;
    marker_buffer.clear();

    header_buffer.clear();
    content_buffer.clear();

    pending_content.clear();
    pending_reasoning.clear();
    pending_tool_calls.clear();

    should_stop = false;
}

void HarmonyParser::start_in_header() {
    // Called when <|start|>role has already been decoded (e.g., generation prompt)
    // Start directly in HEADER state waiting for <|channel|> or <|message|>
    reset();
    parse_state = ParseState::HEADER;
}

void HarmonyParser::start_in_final_content() {
    // Called when thinking is disabled - generation prompt includes <|channel|>final<|message|>
    // Start directly in CONTENT state with FINAL channel, ready for model output
    reset();
    parse_state = ParseState::CONTENT;
    current_channel = Channel::FINAL;
}

bool HarmonyParser::process(const std::string& token) {
    for (char c : token) {
        process_char(c);
        if (should_stop) {
            return true;  // Signal to stop generation
        }
    }

    // Flush content buffer for streaming - move accumulated content to pending
    // This ensures content is available via get_content_delta() after each token
    // BUT: Don't flush if tool_recipient is set - that's a tool call, keep for parsing
    // Note: Tool calls can appear in both analysis and commentary channels
    // (per vLLM and llama.cpp implementations)
    if (parse_state == ParseState::CONTENT && !content_buffer.empty()) {
        if (!tool_recipient.empty()) {
            // Tool call in progress - keep content_buffer for tool call parsing
        } else if (current_channel == Channel::ANALYSIS) {
            pending_reasoning += content_buffer;
            content_buffer.clear();
        } else if (current_channel == Channel::FINAL) {
            pending_content += content_buffer;
            content_buffer.clear();
        }
        // COMMENTARY without tool_recipient: keep in content_buffer (might be preamble)
    }

    return false;
}

void HarmonyParser::process_char(char c) {
    switch (char_state) {
        case CharState::NORMAL:
            if (c == '<') {
                char_state = CharState::SAW_LT;
            } else {
                // Regular content - route based on parse state and channel
                switch (parse_state) {
                    case ParseState::EXPECT_START:
                        // Content before <|start|> is ignored
                        break;
                    case ParseState::HEADER:
                        header_buffer += c;
                        break;
                    case ParseState::CONTENT:
                        content_buffer += c;
                        break;
                }
            }
            break;

        case CharState::SAW_LT:
            if (c == '|') {
                char_state = CharState::SAW_LT_PIPE;
                marker_buffer.clear();
            } else {
                // Not a marker - emit the '<' and this character
                switch (parse_state) {
                    case ParseState::EXPECT_START:
                        break;
                    case ParseState::HEADER:
                        header_buffer += '<';
                        header_buffer += c;
                        break;
                    case ParseState::CONTENT:
                        content_buffer += '<';
                        content_buffer += c;
                        break;
                }
                char_state = CharState::NORMAL;
            }
            break;

        case CharState::SAW_LT_PIPE:
            if (c == '|') {
                // Saw "<||" - handle as content
                switch (parse_state) {
                    case ParseState::EXPECT_START:
                        break;
                    case ParseState::HEADER:
                        header_buffer += "<|";
                        header_buffer += c;
                        break;
                    case ParseState::CONTENT:
                        content_buffer += "<|";
                        content_buffer += c;
                        break;
                }
                char_state = CharState::NORMAL;
            } else if (c == '>') {
                // Complete marker: <|marker_buffer|>
                handle_marker(marker_buffer);
                marker_buffer.clear();
                char_state = CharState::NORMAL;
            } else {
                // Continue accumulating marker name
                marker_buffer += c;
                char_state = CharState::MATCHING_MARKER;
            }
            break;

        case CharState::MATCHING_MARKER:
            if (c == '|') {
                // Expect '>' next
                char_state = CharState::SAW_LT_PIPE;  // Re-use state to check for '>'
            } else if (c == '>') {
                // Complete marker: <|marker_buffer|>
                handle_marker(marker_buffer);
                marker_buffer.clear();
                char_state = CharState::NORMAL;
            } else {
                // Continue accumulating
                marker_buffer += c;
            }
            break;
    }
}

void HarmonyParser::handle_marker(const std::string& marker_name) {
    if (marker_name == "start") {
        // <|start|> - Begin new message
        // Reset for new message
        header_buffer.clear();
        content_buffer.clear();
        current_channel = Channel::NONE;
        tool_recipient.clear();
        parse_state = ParseState::HEADER;

    } else if (marker_name == "message") {
        // <|message|> - End of header, start of content
        extract_channel_from_header();
        content_buffer.clear();
        parse_state = ParseState::CONTENT;

    } else if (marker_name == "channel") {
        // <|channel|> marker - next content until next marker is channel name
        // Channel name will be extracted in extract_channel_from_header()
        if (parse_state == ParseState::CONTENT) {
            // Model is switching channels mid-stream (e.g., after injected final channel)
            // Flush current content to appropriate pending based on current channel
            if (current_channel == Channel::ANALYSIS) {
                pending_reasoning += content_buffer;
            } else if (current_channel == Channel::FINAL || current_channel == Channel::COMMENTARY) {
                if (current_channel == Channel::COMMENTARY && !tool_recipient.empty()) {
                    parse_tool_call_from_buffer();
                } else {
                    pending_content += content_buffer;
                }
            }
            content_buffer.clear();
            // Transition to HEADER state to capture new channel
            parse_state = ParseState::HEADER;
            header_buffer.clear();
        } else if (parse_state == ParseState::EXPECT_START) {
            // Model skipped <|start|> and went directly to <|channel|>
            // Treat as starting a new message
            parse_state = ParseState::HEADER;
            header_buffer.clear();
            current_channel = Channel::NONE;
            tool_recipient.clear();
        }
        // Add channel marker to header buffer
        if (parse_state == ParseState::HEADER) {
            header_buffer += "<|channel|>";
        }

    } else if (marker_name == "end") {
        // <|end|> - End of message, may continue with next message
        // Handle malformed message: if still in HEADER state, treat header_buffer as content
        if (parse_state == ParseState::HEADER && !header_buffer.empty()) {
            // Model output text without proper <|channel|><|message|> markers
            // Treat as FINAL channel content (fallback for malformed messages)
            pending_content += header_buffer;
            header_buffer.clear();
        }
        // Flush content to appropriate output
        if (current_channel == Channel::ANALYSIS) {
            pending_reasoning += content_buffer;
            // Add newline after reasoning so content starts on new line
            if (!content_buffer.empty()) {
                pending_reasoning += "\n";
            }
        } else if (current_channel == Channel::FINAL || current_channel == Channel::COMMENTARY) {
            // Check for tool call in commentary channel
            if (current_channel == Channel::COMMENTARY && !tool_recipient.empty()) {
                parse_tool_call_from_buffer();
            } else {
                pending_content += content_buffer;
            }
        }
        content_buffer.clear();
        parse_state = ParseState::EXPECT_START;

    } else if (marker_name == "return") {
        // <|return|> - Stop generation
        // Handle malformed message: if still in HEADER state, treat header_buffer as content
        if (parse_state == ParseState::HEADER && !header_buffer.empty()) {
            pending_content += header_buffer;
            header_buffer.clear();
        }
        // Flush any remaining content
        if (current_channel == Channel::ANALYSIS) {
            pending_reasoning += content_buffer;
        } else if (current_channel == Channel::FINAL || current_channel == Channel::COMMENTARY) {
            pending_content += content_buffer;
        }
        content_buffer.clear();
        parse_state = ParseState::EXPECT_START;
        should_stop = true;

    } else if (marker_name == "call") {
        // <|call|> - Tool call, stop generation
        // The tool call should have been parsed from content
        dout(2) << "HarmonyParser: <|call|> detected, tool_recipient='" << tool_recipient
                << "', content_buffer length=" << content_buffer.length() << std::endl;
        if (!tool_recipient.empty()) {
            parse_tool_call_from_buffer();
        } else {
            dout(1) << "HarmonyParser: WARNING - <|call|> with empty tool_recipient, tool call dropped!" << std::endl;
            dout(2) << "HarmonyParser: header_buffer was: " << header_buffer << std::endl;
            dout(2) << "HarmonyParser: content_buffer was: " << content_buffer.substr(0, 200) << std::endl;
        }
        content_buffer.clear();
        parse_state = ParseState::EXPECT_START;
        should_stop = true;

    } else if (marker_name == "constrain") {
        // <|constrain|> - Constraint marker (ignored)
        if (parse_state == ParseState::HEADER) {
            header_buffer += "<|constrain|>";
        }

    } else {
        // Unknown marker - treat as content
        std::string marker_text = "<|" + marker_name + "|>";
        switch (parse_state) {
            case ParseState::EXPECT_START:
                break;
            case ParseState::HEADER:
                header_buffer += marker_text;
                break;
            case ParseState::CONTENT:
                content_buffer += marker_text;
                break;
        }
    }
}

void HarmonyParser::extract_channel_from_header() {
    // Header format examples:
    // "assistant<|channel|>analysis"
    // "assistant<|channel|>final"
    // "assistant<|channel|>commentary to=functions.tool_name"
    // "assistant<|channel|>analysis to=functions.bash code"

    dout(2) << "HarmonyParser: extract_channel_from_header: '" << header_buffer << "'" << std::endl;

    // Check for tool recipient pattern: "to=functions.X"
    static const std::regex recipient_regex("to=functions\\.([^<\\s]+)");
    std::smatch match;
    if (std::regex_search(header_buffer, match, recipient_regex)) {
        tool_recipient = match[1].str();
        dout(2) << "HarmonyParser: found tool_recipient='" << tool_recipient << "'" << std::endl;
    } else {
        tool_recipient.clear();
    }

    // Check for built-in tool recipients that we don't support
    static const std::regex builtin_regex("to=(browser\\.[^<\\s]+|python)");
    if (std::regex_search(header_buffer, match, builtin_regex)) {
        std::string builtin_tool = match[1].str();
        dout(0) << "WARNING: Model tried to call built-in tool '" << builtin_tool
                << "' which is not supported. Tool call will be ignored." << std::endl;
        // Don't set tool_recipient - this will cause the call to be ignored
    }

    // Extract constraint type (json or code) - appears at end of header
    static const std::regex constraint_regex("\\b(json|code)\\s*$");
    if (std::regex_search(header_buffer, match, constraint_regex)) {
        constraint_type = match[1].str();
        dout(2) << "HarmonyParser: found constraint_type='" << constraint_type << "'" << std::endl;
    } else {
        constraint_type = "json";  // default
    }

    // Extract channel type
    if (header_buffer.find("<|channel|>analysis") != std::string::npos) {
        current_channel = Channel::ANALYSIS;
    } else if (header_buffer.find("<|channel|>final") != std::string::npos) {
        current_channel = Channel::FINAL;
    } else if (header_buffer.find("<|channel|>commentary") != std::string::npos) {
        current_channel = Channel::COMMENTARY;
    } else {
        // Default to FINAL for unknown channels
        current_channel = Channel::FINAL;
    }
}

void HarmonyParser::parse_tool_call_from_buffer() {
    if (tool_recipient.empty() || content_buffer.empty()) {
        // No tool call to parse - don't treat as content (it was meant to be a tool call)
        return;
    }

    // Try to extract JSON arguments from content
    std::string args = content_buffer;

    // Trim whitespace
    size_t start = args.find_first_not_of(" \t\n\r");
    size_t end = args.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
        args = args.substr(start, end - start + 1);
    }

    // If not valid JSON, wrap as empty object
    if (args.empty() || args[0] != '{') {
        args = "{}";
    }

    // Generate tool call ID
    static int tool_call_counter = 0;
    std::string tool_id = "call_" + std::to_string(++tool_call_counter);

    pending_tool_calls.push_back({tool_recipient, args, tool_id});
}

std::string HarmonyParser::get_content_delta() {
    std::string delta = std::move(pending_content);
    pending_content.clear();
    return delta;
}

std::string HarmonyParser::get_reasoning_delta() {
    std::string delta = std::move(pending_reasoning);
    pending_reasoning.clear();
    return delta;
}

std::vector<ToolCall> HarmonyParser::get_tool_calls() {
    std::vector<ToolCall> calls = std::move(pending_tool_calls);
    pending_tool_calls.clear();
    return calls;
}

bool HarmonyParser::has_pending_content() const {
    return !marker_buffer.empty() || !content_buffer.empty() || !header_buffer.empty();
}

void HarmonyParser::flush() {
    // Flush any incomplete marker detection as content
    // This handles cases where generation ends mid-marker (e.g., "Hello<" or "Hello<|retu")
    if (char_state != CharState::NORMAL && parse_state == ParseState::CONTENT) {
        std::string incomplete_marker;
        switch (char_state) {
            case CharState::SAW_LT:
                incomplete_marker = "<";
                break;
            case CharState::SAW_LT_PIPE:
            case CharState::MATCHING_MARKER:
                incomplete_marker = "<|" + marker_buffer;
                break;
            default:
                break;
        }
        if (!incomplete_marker.empty()) {
            if (current_channel == Channel::ANALYSIS) {
                pending_reasoning += incomplete_marker;
            } else {
                pending_content += incomplete_marker;
            }
        }
    }

    // Flush any partial content buffer
    if (parse_state == ParseState::CONTENT && !content_buffer.empty()) {
        if (current_channel == Channel::ANALYSIS) {
            pending_reasoning += content_buffer;
        } else {
            pending_content += content_buffer;
        }
        content_buffer.clear();
    }

    // Reset to initial state
    parse_state = ParseState::EXPECT_START;
    char_state = CharState::NORMAL;
    marker_buffer.clear();
}

} // namespace StreamParser
