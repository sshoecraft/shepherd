
#pragma once

#include <string>
#include <functional>

namespace ChannelParsing {

// Event types emitted by the channel parser
enum class EventType {
    CONTENT,     // Final channel content (shown to user)
    THINKING,    // Analysis channel content (reasoning)
    TOOL_CALL,   // Tool call from commentary channel
    PREAMBLE     // Commentary channel text (not a tool call)
};

// Callback for parser events
// Returns true to continue, false to cancel
using EventCallback = std::function<bool(EventType type,
                                          const std::string& content,
                                          const std::string& tool_name,
                                          const std::string& tool_args)>;

// Stateful channel parser for GPT-OSS format
// Instantiate per-request for isolation
class ChannelParser {
public:
    // Configuration
    struct Config {
        bool has_channels = false;       // From ChatTemplateCaps
        bool include_reasoning = false;  // Show analysis channel as THINKING
        std::string channel_start = "<|channel|>";
        std::string message_start = "<|message|>";
        std::string channel_end = "<|end|>";
        std::string turn_start = "<|start|>";  // Signals new turn - stop generation
    };

    explicit ChannelParser(const Config& config);

    // Process a delta of text and emit events
    // For non-channel models: immediate pass-through
    // For channel models: buffer and parse channels
    // Returns true to continue, false to stop generation (e.g., new turn detected)
    bool process(const std::string& delta, EventCallback callback);

    // Flush any remaining buffered content at end of stream
    void flush(EventCallback callback);

private:
    enum class Channel {
        NONE,        // Before first channel marker
        ANALYSIS,    // <|channel|>analysis - reasoning
        COMMENTARY,  // <|channel|>commentary - tool calls or preamble
        FINAL        // <|channel|>final - user-visible content
    };

    Config config_;
    Channel current_channel_ = Channel::NONE;
    std::string buffer_;
    bool in_message_ = false;  // True after seeing <|message|>
    bool seen_final_channel_ = false;  // True after we've exited a final channel

    // Parse channel type from buffer after <|channel|>
    Channel parse_channel_type(const std::string& type_str);

    // Check for and extract tool recipient from commentary
    // Returns tool name if "to=functions.tool_name" found, empty otherwise
    std::string extract_tool_recipient(const std::string& content);

    // Emit buffered content based on current channel
    void emit_channel_content(const std::string& content, EventCallback callback);
};

} // namespace ChannelParsing
