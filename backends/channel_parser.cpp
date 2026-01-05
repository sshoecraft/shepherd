
#include "channel_parser.h"
#include "shepherd.h"
#include <algorithm>

namespace ChannelParsing {

ChannelParser::ChannelParser(const Config& config)
    : cfg(config) {
}

bool ChannelParser::process(const std::string& delta, EventCallback callback) {
    // Non-channel models: immediate pass-through
    if (!cfg.has_channels) {
        callback(EventType::CONTENT, delta, "", "");
        return true;
    }

    // Append delta to buffer
    buffer += delta;

    // Check for turn start marker (e.g., <|start|>) - signals new turn, stop generation
    // Only stop if we've already seen a final channel - the first <|start|>assistant after
    // analysis is expected format, not a new turn
    if (!cfg.turn_start.empty() && seen_final_channel) {
        size_t turn_pos = buffer.find(cfg.turn_start);

        // Also check for partial/broken turn_start markers (e.g., "tart|>" from broken "<|start|>")
        // This handles cases where llama.cpp's special token filtering partially renders tokens
        // The token might be split as "<|s" + "tart|>" with first part suppressed
        if (turn_pos == std::string::npos && cfg.turn_start.length() > 4) {
            // Check for various partial suffixes
            std::vector<std::string> partial_markers;
            // For "<|start|>", check: "start|>", "tart|>", "art|>", "|>"
            for (size_t skip = 2; skip < cfg.turn_start.length() - 2; skip++) {
                partial_markers.push_back(cfg.turn_start.substr(skip));
            }
            for (const auto& partial : partial_markers) {
                turn_pos = buffer.find(partial);
                if (turn_pos != std::string::npos) {
                    dout(1) << "ChannelParser: detected partial turn_start marker [" << partial << "]" << std::endl;
                    break;
                }
            }
        }

        if (turn_pos != std::string::npos) {
            dout(1) << "ChannelParser: detected turn_start marker after final channel, signaling stop" << std::endl;
            // Emit any content before the turn marker
            if (turn_pos > 0 && current_channel == Channel::FINAL) {
                std::string content = buffer.substr(0, turn_pos);
                emit_channel_content(content, callback);
            }
            buffer.clear();
            return false;  // Signal to stop generation
        }
    }

    // Process buffer looking for channel markers
    while (!buffer.empty()) {
        if (current_channel == Channel::NONE) {
            // Looking for <|channel|>type<|message|>
            size_t channel_pos = buffer.find(cfg.channel_start);
            if (channel_pos == std::string::npos) {
                // No channel marker yet - keep buffering
                // But limit buffer size to prevent memory issues
                if (buffer.size() > 16384) {
                    dout(1) << "ChannelParser: buffer overflow, discarding" << std::endl;
                    buffer.clear();
                }
                return true;
            }

            // Discard anything before the channel marker
            if (channel_pos > 0) {
                buffer = buffer.substr(channel_pos);
            }

            // Find the channel type (between <|channel|> and <|message|>)
            size_t type_start = cfg.channel_start.length();
            size_t message_pos = buffer.find(cfg.message_start, type_start);
            if (message_pos == std::string::npos) {
                // Haven't received full channel header yet
                return true;
            }

            // Extract and parse channel type
            std::string type_str = buffer.substr(type_start, message_pos - type_start);
            current_channel = parse_channel_type(type_str);
            channel_header = type_str;  // Store full header for tool name extraction
            in_message = true;

            dout(2) << "ChannelParser: entering channel [" << type_str << "]" << std::endl;

            // Remove the header from buffer
            buffer = buffer.substr(message_pos + cfg.message_start.length());
        }

        // We're inside a channel - look for <|end|>
        size_t end_pos = buffer.find(cfg.channel_end);
        if (end_pos == std::string::npos) {
            // No end marker yet - check for <|start|> which signals new turn
            if (!cfg.turn_start.empty()) {
                size_t stop_pos = buffer.find(cfg.turn_start);
                if (stop_pos != std::string::npos) {
                    // Found <|start|> - emit content before it and stop
                    if (stop_pos > 0) {
                        emit_channel_content(buffer.substr(0, stop_pos), callback);
                    }
                    buffer.clear();
                    dout(1) << "ChannelParser: detected turn_start, stopping" << std::endl;
                    return false;
                }
            }

            // Stream content as it arrives (don't buffer excessively)
            // BUT keep potential partial markers at the end of buffer
            if (!buffer.empty()) {
                size_t safe_len = buffer.length();

                // Check for partial markers at the end - don't stream them yet
                std::vector<std::string> markers = {cfg.channel_end, cfg.turn_start, cfg.channel_start};
                for (const auto& marker : markers) {
                    if (marker.empty()) continue;
                    // Check if buffer ends with any prefix of this marker
                    for (size_t prefix_len = 1; prefix_len < marker.length() && prefix_len <= buffer.length(); prefix_len++) {
                        if (buffer.compare(buffer.length() - prefix_len, prefix_len, marker, 0, prefix_len) == 0) {
                            safe_len = std::min(safe_len, buffer.length() - prefix_len);
                            break;
                        }
                    }
                }

                // Emit content that's safe (not a potential partial marker)
                if (safe_len > 0) {
                    emit_channel_content(buffer.substr(0, safe_len), callback);
                }
                buffer = buffer.substr(safe_len);
            }
            return true;
        }

        // Found end marker - emit content up to it
        std::string content = buffer.substr(0, end_pos);
        if (!content.empty()) {
            emit_channel_content(content, callback);
        }

        dout(2) << "ChannelParser: exiting channel" << std::endl;

        // Track if we've seen a final channel (for stop detection)
        if (current_channel == Channel::FINAL) {
            seen_final_channel = true;
        }

        // Move past the end marker
        buffer = buffer.substr(end_pos + cfg.channel_end.length());
        current_channel = Channel::NONE;
        channel_header.clear();
        in_message = false;
    }
    return true;
}

void ChannelParser::flush(EventCallback callback) {
    // Emit any remaining buffered content
    if (!buffer.empty() && current_channel != Channel::NONE) {
        emit_channel_content(buffer, callback);
        buffer.clear();
    }
    current_channel = Channel::NONE;
    channel_header.clear();
    in_message = false;
}

ChannelParser::Channel ChannelParser::parse_channel_type(const std::string& type_str) {
    // Check for prefix match - channel types may have attributes like "commentary to=functions.X"
    if (type_str == "analysis" || type_str.rfind("analysis ", 0) == 0) {
        return Channel::ANALYSIS;
    } else if (type_str == "commentary" || type_str.rfind("commentary ", 0) == 0) {
        return Channel::COMMENTARY;
    } else if (type_str == "final" || type_str.rfind("final ", 0) == 0) {
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
    switch (current_channel) {
        case Channel::ANALYSIS:
            // Reasoning/thinking - emit only if include_reasoning is enabled
            if (cfg.include_reasoning) {
                callback(EventType::THINKING, content, "", "");
            }
            // Otherwise suppress
            break;

        case Channel::COMMENTARY: {
            // Check if this is a tool call with explicit "to=functions.X" format in header
            std::string tool_name = extract_tool_recipient(channel_header);
            if (!tool_name.empty()) {
                // Explicit tool call format - content is the args
                callback(EventType::TOOL_CALL, "", tool_name, content);
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
