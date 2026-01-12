// Harmony GPT-OSS format parser
// Ported from llama.cpp common/chat.cpp (common_chat_parse_gpt_oss)
// Backend-agnostic implementation for parsing GPT-OSS channel-based responses

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <regex>
#include <stdexcept>

namespace Harmony {

// Tool call structure (matches llama.cpp common_chat_tool_call)
struct ToolCall {
    std::string name;
    std::string arguments;  // JSON string
    std::string id;

    bool operator==(const ToolCall& other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
};

// Parsed message result (matches llama.cpp common_chat_msg)
struct ParsedMessage {
    std::string role = "assistant";
    std::string content;           // From final/commentary channels
    std::string reasoning_content; // From analysis channel
    std::vector<ToolCall> tool_calls;

    bool empty() const {
        return content.empty() && reasoning_content.empty() && tool_calls.empty();
    }
};

// Parsing options
struct ParseOptions {
    bool is_partial = false;         // Streaming mode (incomplete input)
    bool parse_tool_calls = true;    // Extract tool calls from commentary channel
};

// Exception for incomplete parsing (streaming mode)
class PartialException : public std::runtime_error {
public:
    explicit PartialException(const std::string& msg) : std::runtime_error(msg) {}
};

// String range for regex matches
struct StringRange {
    size_t begin;
    size_t end;

    StringRange(size_t b, size_t e) : begin(b), end(e) {
        if (b > e) throw std::runtime_error("Invalid range");
    }

    bool empty() const { return begin == end; }
};

// Regex match result
struct RegexMatch {
    enum Type { NONE, PARTIAL, FULL } type = NONE;
    std::vector<StringRange> groups;
};

// Regex wrapper with partial matching support
// Ported from llama.cpp common/regex-partial.cpp
class Regex {
    std::string pattern;
    std::regex rx;
    std::regex rx_reversed_partial;

public:
    explicit Regex(const std::string& pat);

    RegexMatch search(const std::string& input, size_t pos, bool as_match = false) const;

    const std::string& str() const { return pattern; }
};

// Internal parser builder class
// Ported from llama.cpp common/chat-parser.cpp
class MsgParser {
    std::string input;
    bool partial;
    bool parse_tools;
    std::string healing_marker;
    size_t position = 0;
    ParsedMessage result;

public:
    MsgParser(const std::string& in, const ParseOptions& opts);

    const std::string& get_input() const { return input; }
    size_t pos() const { return position; }
    bool is_partial() const { return partial; }
    bool parse_tool_calls() const { return parse_tools; }
    const ParsedMessage& get_result() const { return result; }

    void move_to(size_t p);
    std::string str(const StringRange& rng) const;

    // Content builders
    void add_content(const std::string& s);
    void add_reasoning_content(const std::string& s);
    bool add_tool_call(const std::string& name, const std::string& id, const std::string& args);

    // Parsing helpers
    std::string consume_rest();
    bool try_consume_literal(const std::string& literal);

    struct FindResult {
        std::string prelude;
        std::vector<StringRange> groups;
    };

    std::optional<FindResult> try_find_literal(const std::string& literal);
    std::optional<FindResult> try_find_regex(const Regex& rx, size_t from = std::string::npos, bool add_prelude_to_content = true);

    // JSON parsing with partial support
    struct JsonResult {
        std::string json_str;
        bool is_partial;
    };
    std::optional<JsonResult> try_consume_json();

    void finish();
    void clear_tools();
};

// Main parser class
class Parser {
public:
    Parser() = default;

    // Parse complete or partial GPT-OSS response
    ParsedMessage parse(const std::string& input, const ParseOptions& opts = {});

    // Streaming API
    void reset();
    void feed(const std::string& text);
    ParsedMessage get_partial_result();

    // Get content delta since last call
    std::string consume_content_delta();
    std::string consume_reasoning_delta();

    bool has_content_delta() const { return last_content_length < accumulated_result.content.length(); }
    bool has_reasoning_delta() const { return last_reasoning_length < accumulated_result.reasoning_content.length(); }

private:
    std::string accumulated;
    ParsedMessage accumulated_result;
    size_t last_content_length = 0;
    size_t last_reasoning_length = 0;
};

// Streaming callback interface (for compatibility with old API)
enum class EventType {
    CONTENT,      // Regular content from final channel
    THINKING,     // Content from analysis channel (reasoning)
    TOOL_CALL,    // Tool call from commentary channel
    PREAMBLE      // Content from commentary channel (not a tool call)
};

using EventCallback = std::function<bool(
    EventType event,
    const std::string& content,
    const std::string& tool_name,
    const std::string& tool_args
)>;

// Legacy streaming parser wrapper
class StreamingParser {
public:
    StreamingParser() = default;

    // Process text chunk, emit events via callback
    bool process(const std::string& text, const EventCallback& callback = nullptr);

    // Flush remaining content
    void flush(const EventCallback& callback = nullptr);

    // Reset for new response
    void reset();

    // Get accumulated content
    std::string consume_content_delta();
    std::string current_channel() const;
    std::string current_recipient() const;

private:
    Parser parser;
    std::string last_channel;
    std::string last_recipient;
    size_t last_tool_call_count = 0;
};

} // namespace Harmony
