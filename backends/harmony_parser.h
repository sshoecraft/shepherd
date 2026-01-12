// HarmonyParser - O(n) streaming parser for GPT-OSS harmony format
// Character-by-character state machine for channel-based parsing

#pragma once

#include "parser.h"
#include <string>
#include <vector>

namespace StreamParser {

class HarmonyParser : public Parser {
public:
    HarmonyParser() = default;

    // Parser interface
    bool process(const std::string& token) override;
    std::string get_content_delta() override;
    std::string get_reasoning_delta() override;
    std::vector<ToolCall> get_tool_calls() override;
    void reset() override;
    bool has_pending_content() const override;
    void flush() override;

    // Start parser in HEADER state (for when <|start|>role is already decoded)
    void start_in_header();

    // Start parser in CONTENT state with FINAL channel (for when thinking is disabled)
    // Used when generation prompt includes <|channel|>final<|message|>
    void start_in_final_content();

private:
    // High-level parsing state
    enum class ParseState {
        EXPECT_START,   // Waiting for <|start|>
        HEADER,         // In header, waiting for <|message|>
        CONTENT         // In content, waiting for <|end|>, <|return|>, <|call|>
    };

    // Character-level marker detection state
    enum class CharState {
        NORMAL,           // Regular content
        SAW_LT,           // Saw '<'
        SAW_LT_PIPE,      // Saw '<|'
        MATCHING_MARKER   // Accumulating marker name
    };

    // Channel types
    enum class Channel {
        NONE,
        ANALYSIS,     // Reasoning/thinking
        FINAL,        // User-facing content
        COMMENTARY    // May contain tool calls
    };

    // Process a single character
    void process_char(char c);

    // Handle a complete marker (e.g., "start", "end", "message", "channel", "return", "call")
    void handle_marker(const std::string& marker_name);

    // Extract channel name from header buffer
    void extract_channel_from_header();

    // Parse tool call from buffered content
    void parse_tool_call_from_buffer();

    // High-level state
    ParseState parse_state = ParseState::EXPECT_START;
    Channel current_channel = Channel::NONE;
    std::string tool_recipient;  // For "to=functions.X" pattern

    // Character-level state
    CharState char_state = CharState::NORMAL;
    std::string marker_buffer;  // Partial marker being accumulated (e.g., "star" -> "start")

    // Content buffers
    std::string header_buffer;  // Header content (between <|start|> and <|message|>)
    std::string content_buffer; // Content (between <|message|> and <|end|>/<|return|>/<|call|>)

    // Output deltas
    std::string pending_content;
    std::string pending_reasoning;
    std::vector<ToolCall> pending_tool_calls;

    // Stop flag
    bool should_stop = false;
};

} // namespace StreamParser
