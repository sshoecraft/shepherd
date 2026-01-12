// Abstract streaming parser interface for token-by-token processing
// Used by both GenericParser (non-harmony models) and HarmonyParser (GPT-OSS)

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace StreamParser {

// Parsed tool call (common format for all parsers)
struct ToolCall {
    std::string name;
    std::string arguments;  // JSON string
    std::string id;
};

// Abstract base class for streaming parsers
// Processes tokens incrementally and extracts content, reasoning, and tool calls
class Parser {
public:
    virtual ~Parser() = default;

    // Process a token, return true if generation should STOP
    // Called for each token during generation
    virtual bool process(const std::string& token) = 0;

    // Get content delta since last call (user-facing response)
    virtual std::string get_content_delta() = 0;

    // Get reasoning delta since last call (thinking/analysis)
    virtual std::string get_reasoning_delta() = 0;

    // Get completed tool calls since last call
    virtual std::vector<ToolCall> get_tool_calls() = 0;

    // Reset parser state for new generation
    virtual void reset() = 0;

    // Check if parser has pending content (for flush operations)
    virtual bool has_pending_content() const { return false; }

    // Flush any buffered content (call at end of generation)
    virtual void flush() {}
};

// Factory function to create appropriate parser based on model capabilities
// Implemented in generic_parser.cpp / harmony_parser.cpp
std::unique_ptr<Parser> create_parser(
    bool has_harmony_channels,
    const std::vector<std::string>& tool_start_markers = {},
    const std::vector<std::string>& tool_end_markers = {},
    const std::vector<std::string>& thinking_start_markers = {},
    const std::vector<std::string>& thinking_end_markers = {}
);

} // namespace StreamParser
