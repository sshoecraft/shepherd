// Harmony GPT-OSS format parser implementation
// Ported from llama.cpp common/chat.cpp (common_chat_parse_gpt_oss)

#include "harmony.h"
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cctype>

namespace Harmony {

// Helper: find partial stop in string
static size_t string_find_partial_stop(const std::string& str, const std::string& stop) {
    size_t str_len = str.length();
    size_t stop_len = stop.length();

    for (size_t i = 1; i < stop_len && i <= str_len; i++) {
        if (str.compare(str_len - i, i, stop, 0, i) == 0) {
            return str_len - i;
        }
    }
    return std::string::npos;
}

// Helper: split string
static std::vector<std::string> string_split(const std::string& str, const std::string& delim) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = str.find(delim);
    while (end != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + delim.length();
        end = str.find(delim, start);
    }
    result.push_back(str.substr(start));
    return result;
}

// Helper: join strings
static std::string string_join(const std::vector<std::string>& parts, const std::string& sep) {
    std::string result;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i > 0) result += sep;
        result += parts[i];
    }
    return result;
}

// Convert regex to reversed partial regex for partial matching
// Ported from llama.cpp regex-partial.cpp
static std::string regex_to_reversed_partial_regex(const std::string& pattern) {
    auto it = pattern.begin();
    const auto end = pattern.end();

    std::function<std::string()> process = [&]() {
        std::vector<std::vector<std::string>> alternatives(1);
        std::vector<std::string>* sequence = &alternatives.back();

        while (it != end) {
            if (*it == '[') {
                auto start = it;
                ++it;
                while (it != end) {
                    if ((*it == '\\') && (++it != end)) {
                        ++it;
                    } else if ((it != end) && (*it == ']')) {
                        break;
                    } else {
                        ++it;
                    }
                }
                if (it == end) {
                    throw std::runtime_error("Unmatched '[' in pattern");
                }
                ++it;
                sequence->push_back(std::string(start, it));
            } else if (*it == '*' || *it == '?' || *it == '+') {
                if (sequence->empty()) {
                    throw std::runtime_error("Quantifier without preceding element");
                }
                sequence->back() += *it;
                auto is_star = *it == '*';
                ++it;
                if (is_star) {
                    if (it != end && *it == '?') {
                        ++it;
                    }
                }
            } else if (*it == '{') {
                if (sequence->empty()) {
                    throw std::runtime_error("Repetition without preceding element");
                }
                ++it;
                auto start = it;
                while (it != end && *it != '}') {
                    ++it;
                }
                if (it == end) {
                    throw std::runtime_error("Unmatched '{' in pattern");
                }
                auto parts = string_split(std::string(start, it), ",");
                ++it;
                if (parts.size() > 2) {
                    throw std::runtime_error("Invalid repetition range in pattern");
                }

                auto parseOptInt = [&](const std::string& s, const std::optional<int>& def = std::nullopt) -> std::optional<int> {
                    if (s.empty()) {
                        return def;
                    }
                    return std::stoi(s);
                };
                auto min = parseOptInt(parts[0], 0);
                auto max = parts.size() == 1 ? min : parseOptInt(parts[1]);
                if (min && max && *max < *min) {
                    throw std::runtime_error("Invalid repetition range in pattern");
                }
                auto part = sequence->back();
                sequence->pop_back();
                for (int i = 0; i < *min; i++) {
                    sequence->push_back(part);
                }
                if (max) {
                    for (int i = *min; i < *max; i++) {
                        sequence->push_back(part + "?");
                    }
                } else {
                    sequence->push_back(part + "*");
                }
            } else if (*it == '(') {
                ++it;
                if (it != end && *it == '?' && (it + 1 != end) && *(it + 1) == ':') {
                    it += 2;
                }
                auto sub = process();
                if (*it != ')') {
                    throw std::runtime_error("Unmatched '(' in pattern");
                }
                ++it;
                auto& part = sequence->emplace_back("(?:");
                part += sub;
                part += ")";
            } else if (*it == ')') {
                break;
            } else if (*it == '|') {
                ++it;
                alternatives.emplace_back();
                sequence = &alternatives.back();
            } else if (*it == '\\' && (++it != end)) {
                auto str = std::string("\\") + *it;
                sequence->push_back(str);
                ++it;
            } else if (it != end) {
                sequence->push_back(std::string(1, *it));
                ++it;
            }
        }

        std::vector<std::string> res_alts;
        for (const auto& parts : alternatives) {
            auto& res = res_alts.emplace_back();
            for (size_t i = 0; i < parts.size() - 1; i++) {
                res += "(?:";
            }
            for (auto pit = parts.rbegin(); pit != parts.rend(); ++pit) {
                res += *pit;
                if (pit != parts.rend() - 1) {
                    res += ")?";
                }
            }
        }
        return string_join(res_alts, "|");
    };
    auto res = process();
    if (it != end) {
        throw std::runtime_error("Unmatched '(' in pattern");
    }

    return "(" + res + ")[\\s\\S]*";
}

// Regex class implementation
Regex::Regex(const std::string& pat)
    : pattern(pat)
    , rx(pat)
    , rx_reversed_partial(regex_to_reversed_partial_regex(pat)) {}

RegexMatch Regex::search(const std::string& input, size_t pos, bool as_match) const {
    std::smatch match;
    if (pos > input.size()) {
        throw std::runtime_error("Position out of bounds");
    }
    auto start = input.begin() + pos;
    auto found = as_match
        ? std::regex_match(start, input.end(), match, rx)
        : std::regex_search(start, input.end(), match, rx);

    if (found) {
        RegexMatch res;
        res.type = RegexMatch::FULL;
        for (size_t i = 0; i < match.size(); ++i) {
            auto begin = pos + match.position(i);
            res.groups.emplace_back(begin, begin + match.length(i));
        }
        return res;
    }

    std::match_results<std::string::const_reverse_iterator> srmatch;
    if (std::regex_match(input.rbegin(), input.rend() - pos, srmatch, rx_reversed_partial)) {
        auto group = srmatch[1].str();
        if (group.length() != 0) {
            auto it = srmatch[1].second.base();
            if ((!as_match) || it == input.begin()) {
                RegexMatch res;
                res.type = RegexMatch::PARTIAL;
                const size_t begin = std::distance(input.begin(), it);
                const size_t end = input.size();
                if (begin != std::string::npos && end != std::string::npos && begin <= end) {
                    res.groups.push_back({begin, end});
                    return res;
                }
            }
        }
    }
    return {};
}

// MsgParser implementation
MsgParser::MsgParser(const std::string& in, const ParseOptions& opts)
    : input(in)
    , partial(opts.is_partial)
    , parse_tools(opts.parse_tool_calls)
{
    result.role = "assistant";

    // Generate unique healing marker
    while (true) {
        std::string id = std::to_string(std::rand());
        if (input.find(id) == std::string::npos) {
            healing_marker = id;
            break;
        }
    }
}

void MsgParser::move_to(size_t p) {
    if (p > input.size()) {
        throw std::runtime_error("Invalid position");
    }
    position = p;
}

std::string MsgParser::str(const StringRange& rng) const {
    return input.substr(rng.begin, rng.end - rng.begin);
}

void MsgParser::add_content(const std::string& s) {
    result.content += s;
}

void MsgParser::add_reasoning_content(const std::string& s) {
    result.reasoning_content += s;
}

bool MsgParser::add_tool_call(const std::string& name, const std::string& id, const std::string& args) {
    if (name.empty()) {
        return false;
    }

    ToolCall tc;
    tc.name = name;
    tc.id = id;
    tc.arguments = args;
    result.tool_calls.push_back(tc);
    return true;
}

std::string MsgParser::consume_rest() {
    auto rest = input.substr(position);
    position = input.size();
    return rest;
}

bool MsgParser::try_consume_literal(const std::string& literal) {
    auto pos = position;
    for (size_t i = 0; i < literal.size(); ++i) {
        if (pos >= input.size()) {
            return false;
        }
        if (input[pos] != literal[i]) {
            return false;
        }
        ++pos;
    }
    position = pos;
    return true;
}

std::optional<MsgParser::FindResult> MsgParser::try_find_literal(const std::string& literal) {
    auto idx = input.find(literal, position);
    if (idx != std::string::npos) {
        FindResult res;
        res.prelude = input.substr(position, idx - position);
        auto end = idx + literal.size();
        res.groups.emplace_back(StringRange{idx, end});
        move_to(end);
        return res;
    }
    if (partial) {
        idx = string_find_partial_stop(input, literal);
        if (idx != std::string::npos && idx >= position) {
            FindResult res;
            res.prelude = input.substr(position, idx - position);
            auto end = input.size();
            res.groups.emplace_back(StringRange{idx, end});
            move_to(end);
            return res;
        }
    }
    return std::nullopt;
}

std::optional<MsgParser::FindResult> MsgParser::try_find_regex(const Regex& rx, size_t from, bool add_prelude_to_content) {
    auto m = rx.search(input, from == std::string::npos ? position : from);
    if (m.type == RegexMatch::NONE) {
        return std::nullopt;
    }
    auto prelude = input.substr(position, m.groups[0].begin - position);
    position = m.groups[0].end;

    if (add_prelude_to_content) {
        add_content(prelude);
    }
    if (m.type == RegexMatch::PARTIAL) {
        if (is_partial()) {
            throw PartialException(rx.str());
        }
        return std::nullopt;
    }

    FindResult res;
    res.prelude = prelude;
    for (const auto& g : m.groups) {
        res.groups.push_back(g);
    }
    return res;
}

std::optional<MsgParser::JsonResult> MsgParser::try_consume_json() {
    // Simple JSON extraction - find balanced braces
    if (position >= input.size() || input[position] != '{') {
        return std::nullopt;
    }

    int brace_count = 0;
    bool in_string = false;
    bool escape_next = false;
    size_t start = position;

    for (size_t i = position; i < input.size(); ++i) {
        char c = input[i];

        if (escape_next) {
            escape_next = false;
            continue;
        }

        if (c == '\\' && in_string) {
            escape_next = true;
            continue;
        }

        if (c == '"') {
            in_string = !in_string;
            continue;
        }

        if (!in_string) {
            if (c == '{') {
                brace_count++;
            } else if (c == '}') {
                brace_count--;
                if (brace_count == 0) {
                    JsonResult res;
                    res.json_str = input.substr(start, i - start + 1);
                    res.is_partial = false;
                    position = i + 1;
                    return res;
                }
            }
        }
    }

    // Incomplete JSON
    if (partial) {
        JsonResult res;
        res.json_str = input.substr(start);
        res.is_partial = true;
        position = input.size();
        return res;
    }

    return std::nullopt;
}

void MsgParser::finish() {
    if (!partial && position != input.size()) {
        // There's remaining content - this is okay, just ignore it
    }
}

void MsgParser::clear_tools() {
    result.tool_calls.clear();
}

// The main GPT-OSS parsing function
// Ported from llama.cpp common/chat.cpp common_chat_parse_gpt_oss
static void parse_gpt_oss(MsgParser& builder) {
    static const std::string constraint = "(?: (<\\|constrain\\|>)?([a-zA-Z0-9_-]+))";
    static const std::string recipient("(?: to=functions\\.([^<\\s]+))");

    static const Regex start_regex("<\\|start\\|>assistant");
    static const Regex analysis_regex("<\\|channel\\|>analysis");
    static const Regex final_regex("<\\|channel\\|>final" + constraint + "?");
    static const Regex preamble_regex("<\\|channel\\|>commentary");
    static const Regex tool_call1_regex(recipient + "<\\|channel\\|>(analysis|commentary)" + constraint + "?");
    static const Regex tool_call2_regex("<\\|channel\\|>(analysis|commentary)" + recipient + constraint + "?");

    // Consume content until end marker: <|end|>, <|return|>, or <|call|>
    // <|end|> = end of message (generation may continue)
    // <|return|> = model done generating (stop token)
    // <|call|> = model wants to call tool (stop token)
    auto consume_end = [&](bool include_end = false) -> std::string {
        // Look for <|end|> marker only - <|return|> and <|call|> are handled
        // at the token level in llamacpp.cpp EOG detection and never make it here
        if (auto res = builder.try_find_literal("<|end|>")) {
            return res->prelude + (include_end ? builder.str(res->groups[0]) : "");
        }
        return builder.consume_rest();
    };

    auto handle_tool_call = [&](const std::string& name) {
        if (auto args = builder.try_consume_json()) {
            if (builder.parse_tool_calls()) {
                if (!builder.add_tool_call(name, "", args->json_str) || args->is_partial) {
                    throw PartialException("incomplete tool call");
                }
            } else if (args->is_partial) {
                throw PartialException("incomplete tool call");
            }
        }
    };

    auto regex_match = [](const Regex& regex, const std::string& input) -> std::optional<RegexMatch> {
        auto match = regex.search(input, 0, true);
        if (match.type == RegexMatch::FULL) {
            return match;
        }
        return std::nullopt;
    };

    do {
        auto header_start_pos = builder.pos();
        auto content_start = builder.try_find_literal("<|message|>");
        if (!content_start) {
            throw PartialException("incomplete header");
        }

        auto header = content_start->prelude;

        // Check for tool call pattern 1: " to=functions.X<|channel|>..."
        if (auto match = regex_match(tool_call1_regex, header)) {
            auto group = match->groups[1];
            auto name = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        // Check for tool call pattern 2: "<|channel|>... to=functions.X..."
        if (auto match = regex_match(tool_call2_regex, header)) {
            auto group = match->groups[2];
            auto name = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        // Check for analysis channel (reasoning)
        if (regex_match(analysis_regex, header)) {
            builder.move_to(header_start_pos);
            // Find <|channel|>analysis<|message|> and consume until <|end|>
            builder.try_find_literal("<|message|>");
            builder.add_reasoning_content(consume_end(false));
            continue;
        }

        // Check for final or commentary (preamble) channel
        if (regex_match(final_regex, header) || regex_match(preamble_regex, header)) {
            builder.add_content(consume_end());
            continue;
        }

        // Unknown header - try to recover by consuming content
        builder.add_content(consume_end());

    } while (builder.try_find_regex(start_regex, std::string::npos, false));

    auto remaining = builder.consume_rest();
    if (!remaining.empty()) {
        // Remaining content after last message - ignore or add to content
    }
}

// Parser implementation
ParsedMessage Parser::parse(const std::string& input, const ParseOptions& opts) {
    MsgParser builder(input, opts);
    try {
        parse_gpt_oss(builder);
    } catch (const PartialException&) {
        if (!opts.is_partial) {
            // Fallback: treat entire input as content
            builder.clear_tools();
            builder.move_to(0);
            builder.add_content(builder.consume_rest());
        }
    }
    builder.finish();
    return builder.get_result();
}

void Parser::reset() {
    accumulated.clear();
    accumulated_result = ParsedMessage{};
    last_content_length = 0;
    last_reasoning_length = 0;
}

void Parser::feed(const std::string& text) {
    accumulated += text;

    // Re-parse the accumulated text
    ParseOptions opts;
    opts.is_partial = true;
    opts.parse_tool_calls = true;

    try {
        accumulated_result = parse(accumulated, opts);
    } catch (const PartialException&) {
        // Partial parse - result may be incomplete
    }
}

ParsedMessage Parser::get_partial_result() {
    return accumulated_result;
}

std::string Parser::consume_content_delta() {
    if (last_content_length >= accumulated_result.content.length()) {
        return "";
    }
    auto delta = accumulated_result.content.substr(last_content_length);
    last_content_length = accumulated_result.content.length();
    return delta;
}

std::string Parser::consume_reasoning_delta() {
    if (last_reasoning_length >= accumulated_result.reasoning_content.length()) {
        return "";
    }
    auto delta = accumulated_result.reasoning_content.substr(last_reasoning_length);
    last_reasoning_length = accumulated_result.reasoning_content.length();
    return delta;
}

// StreamingParser implementation (legacy API compatibility)
bool StreamingParser::process(const std::string& text, const EventCallback& callback) {
    parser.feed(text);

    // Emit content delta
    auto content_delta = parser.consume_content_delta();
    if (!content_delta.empty()) {
        if (!callback(EventType::CONTENT, content_delta, "", "")) {
            return false;
        }
    }

    // Emit reasoning delta
    auto reasoning_delta = parser.consume_reasoning_delta();
    if (!reasoning_delta.empty()) {
        if (!callback(EventType::THINKING, reasoning_delta, "", "")) {
            return false;
        }
    }

    // Emit new tool calls
    auto result = parser.get_partial_result();
    while (last_tool_call_count < result.tool_calls.size()) {
        const auto& tc = result.tool_calls[last_tool_call_count];
        if (!callback(EventType::TOOL_CALL, tc.arguments, tc.name, tc.arguments)) {
            return false;
        }
        last_tool_call_count++;
    }

    return true;
}

void StreamingParser::flush(const EventCallback& callback) {
    // Process any remaining content
    process("", callback);
}

void StreamingParser::reset() {
    parser.reset();
    last_channel.clear();
    last_recipient.clear();
    last_tool_call_count = 0;
}

std::string StreamingParser::consume_content_delta() {
    return parser.consume_content_delta();
}

std::string StreamingParser::current_channel() const {
    return last_channel;
}

std::string StreamingParser::current_recipient() const {
    return last_recipient;
}

} // namespace Harmony
