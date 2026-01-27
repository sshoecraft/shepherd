// GenericParser - Streaming parser for non-harmony models
// Handles tool call detection, thinking tag extraction
// Ported from GpuBackend::output() state machine

#pragma once

#include "parser.h"
#include <string>
#include <vector>

namespace StreamParser {

class GenericParser : public Parser {
public:
    // Constructor with marker configuration
    GenericParser(
        const std::vector<std::string>& tool_start_markers,
        const std::vector<std::string>& tool_end_markers,
        const std::vector<std::string>& thinking_start_markers,
        const std::vector<std::string>& thinking_end_markers
    );

    // Parser interface
    bool process(const std::string& token) override;
    std::string get_content_delta() override;
    std::string get_reasoning_delta() override;
    std::vector<ToolCall> get_tool_calls() override;
    void reset() override;
    bool has_pending_content() const override;
    void flush() override;

private:
    // State machine states (from GpuBackend FilterState)
    enum class State {
        NORMAL,           // Regular content
        DETECTING_TAG,    // Saw '<', checking for marker
        IN_THINKING,      // Inside thinking block
        IN_TOOL_CALL,     // Inside tool call
        CHECKING_CLOSE    // Checking for closing tag
    };

    // Process a single character
    void process_char(char c);

    // Marker matching helpers
    bool matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched = nullptr) const;
    bool could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const;

    // Emit buffered tool call
    void emit_tool_call();

    // State
    State state = State::NORMAL;
    bool in_tool_call = false;
    bool in_thinking = false;
    bool in_code_block = false;      // Inside ``` code block
    int backtick_count = 0;          // Consecutive backticks seen
    int json_brace_depth = 0;

    // Buffers
    std::string tag_buffer;           // Partial tag being detected
    std::string current_tag;          // Current matched tag marker
    std::string buffered_tool_call;   // Tool call content being accumulated
    std::string buffered_thinking;    // Thinking content being accumulated

    // Output deltas (consumed by get_*_delta())
    std::string pending_content;
    std::string pending_reasoning;
    std::vector<ToolCall> pending_tool_calls;

    // Marker configuration
    std::vector<std::string> tool_start_markers;
    std::vector<std::string> tool_end_markers;
    std::vector<std::string> thinking_start_markers;
    std::vector<std::string> thinking_end_markers;
};

} // namespace StreamParser
