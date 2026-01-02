
#include "channel_parser.h"
#include "shepherd.h"
#include <algorithm>

namespace ChannelParsing {

ChannelParser::ChannelParser(const Config& config)
    : config_(config) {
}

bool ChannelParser::process(const std::string& delta, EventCallback callback) {
    // Non-channel models: immediate pass-through
    if (!config_.has_channels) {
        callback(EventType::CONTENT, delta, "", "");
        return true;
    }

    // Append delta to buffer
    buffer_ += delta;

    // Check for turn start marker (e.g., <|start|>) - signals new turn, stop generation
    // Only stop if we've already seen a final channel - the first <|start|>assistant after
    // analysis is expected format, not a new turn
    if (!config_.turn_start.empty() && seen_final_channel_) {
        size_t turn_pos = buffer_.find(config_.turn_start);
        if (turn_pos != std::string::npos) {
            dout(1) << "ChannelParser: detected turn_start marker after final channel, signaling stop" << std::endl;
            // Emit any content before the turn marker
            if (turn_pos > 0 && current_channel_ == Channel::FINAL) {
                std::string content = buffer_.substr(0, turn_pos);
                emit_channel_content(content, callback);
            }
            buffer_.clear();
            return false;  // Signal to stop generation
        }
    }

    // Process buffer looking for channel markers
    while (!buffer_.empty()) {
        if (current_channel_ == Channel::NONE) {
            // Looking for <|channel|>type<|message|>
            size_t channel_pos = buffer_.find(config_.channel_start);
            if (channel_pos == std::string::npos) {
                // No channel marker yet - keep buffering
                // But limit buffer size to prevent memory issues
                if (buffer_.size() > 16384) {
                    dout(1) << "ChannelParser: buffer overflow, discarding" << std::endl;
                    buffer_.clear();
                }
                return true;
            }

            // Discard anything before the channel marker
            if (channel_pos > 0) {
                buffer_ = buffer_.substr(channel_pos);
            }

            // Find the channel type (between <|channel|> and <|message|>)
            size_t type_start = config_.channel_start.length();
            size_t message_pos = buffer_.find(config_.message_start, type_start);
            if (message_pos == std::string::npos) {
                // Haven't received full channel header yet
                return true;
            }

            // Extract and parse channel type
            std::string type_str = buffer_.substr(type_start, message_pos - type_start);
            current_channel_ = parse_channel_type(type_str);
            in_message_ = true;

            dout(2) << "ChannelParser: entering channel [" << type_str << "]" << std::endl;

            // Remove the header from buffer
            buffer_ = buffer_.substr(message_pos + config_.message_start.length());
        }

        // We're inside a channel - look for <|end|>
        size_t end_pos = buffer_.find(config_.channel_end);
        if (end_pos == std::string::npos) {
            // No end marker yet - check for <|start|> which signals new turn
            if (!config_.turn_start.empty()) {
                size_t stop_pos = buffer_.find(config_.turn_start);
                if (stop_pos != std::string::npos) {
                    // Found <|start|> - emit content before it and stop
                    if (stop_pos > 0) {
                        emit_channel_content(buffer_.substr(0, stop_pos), callback);
                    }
                    buffer_.clear();
                    dout(1) << "ChannelParser: detected turn_start, stopping" << std::endl;
                    return false;
                }
            }

            // Stream content as it arrives (don't buffer excessively)
            if (!buffer_.empty()) {
                emit_channel_content(buffer_, callback);
                buffer_.clear();
            }
            return true;
        }

        // Found end marker - emit content up to it
        std::string content = buffer_.substr(0, end_pos);
        if (!content.empty()) {
            emit_channel_content(content, callback);
        }

        dout(2) << "ChannelParser: exiting channel" << std::endl;

        // Track if we've seen a final channel (for stop detection)
        if (current_channel_ == Channel::FINAL) {
            seen_final_channel_ = true;
        }

        // Move past the end marker
        buffer_ = buffer_.substr(end_pos + config_.channel_end.length());
        current_channel_ = Channel::NONE;
        in_message_ = false;
    }
    return true;
}

void ChannelParser::flush(EventCallback callback) {
    // Emit any remaining buffered content
    if (!buffer_.empty() && current_channel_ != Channel::NONE) {
        emit_channel_content(buffer_, callback);
        buffer_.clear();
    }
    current_channel_ = Channel::NONE;
    in_message_ = false;
}

ChannelParser::Channel ChannelParser::parse_channel_type(const std::string& type_str) {
    if (type_str == "analysis") {
        return Channel::ANALYSIS;
    } else if (type_str == "commentary") {
        return Channel::COMMENTARY;
    } else if (type_str == "final") {
        return Channel::FINAL;
    }
    // Unknown channel type - treat as final to show content
    dout(1) << "ChannelParser: unknown channel type [" << type_str << "], treating as final" << std::endl;
    return Channel::FINAL;
}

std::string ChannelParser::extract_tool_recipient(const std::string& content) {
    // Look for "to=functions.tool_name" pattern
    const std::string prefix = "to=functions.";
    size_t pos = content.find(prefix);
    if (pos == std::string::npos) {
        return "";
    }

    // Extract tool name (until whitespace or end)
    size_t start = pos + prefix.length();
    size_t end = start;
    while (end < content.length() && !std::isspace(content[end])) {
        end++;
    }

    return content.substr(start, end - start);
}

void ChannelParser::emit_channel_content(const std::string& content, EventCallback callback) {
    switch (current_channel_) {
        case Channel::ANALYSIS:
            // Reasoning/thinking - emit only if include_reasoning is enabled
            if (config_.include_reasoning) {
                callback(EventType::THINKING, content, "", "");
            }
            // Otherwise suppress
            break;

        case Channel::COMMENTARY: {
            // Check if this is a tool call with explicit "to=functions.X" format
            std::string tool_name = extract_tool_recipient(content);
            if (!tool_name.empty()) {
                // Explicit tool call format - extract args
                size_t json_start = content.find('{');
                std::string args = (json_start != std::string::npos)
                    ? content.substr(json_start)
                    : "";
                callback(EventType::TOOL_CALL, "", tool_name, args);
            } else {
                // Pass through as preamble - output() filter will detect JSON tool calls
                callback(EventType::PREAMBLE, content, "", "");
            }
            break;
        }

        case Channel::FINAL:
            // User-visible content - always emit
            callback(EventType::CONTENT, content, "", "");
            break;

        case Channel::NONE:
            // Should not happen
            break;
    }
}

} // namespace ChannelParsing
