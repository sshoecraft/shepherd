
#include "shepherd.h"
#include "backend.h"

Backend::Backend(size_t ctx_size, Session& session, EventCallback cb)
    : context_size(ctx_size), callback(cb) {
    if (!callback) {
        throw std::invalid_argument("Internal error: callback not passed to backend constructor");
    }
}

// Flush the output buffer to callback
// Returns true to continue, false if callback requested cancellation
bool Backend::flush_filter_buffer() {
    if (output_buffer.empty()) return true;

    CallbackEvent event = in_code_block ? CallbackEvent::CODEBLOCK : CallbackEvent::CONTENT;
    bool cont = callback(event, output_buffer, "", "");
    output_buffer.clear();
    return cont;
}

// Reset filter state (called before new generation)
void Backend::reset_output_state() {
    clear_tool_calls();  // Reset accumulated_tool_calls
    backtick_buffer.clear();
    output_buffer.clear();
    in_code_block = false;
    skip_to_newline = false;
    think_tag_buffer.clear();
    in_think_block = false;
    utf8_incomplete.clear();
}

// Returns the length of valid UTF-8 in text
// If result < text.size(), there's an incomplete multi-byte char at the end
static size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check last 1-3 bytes for incomplete sequences
    for (size_t i = 1; i <= 3 && i <= len; i++) {
        unsigned char c = text[len - i];
        if ((c & 0x80) == 0) {
            // ASCII byte - all previous bytes are complete
            return len;
        } else if ((c & 0xC0) == 0xC0) {
            // Start of multi-byte sequence
            int expected_len;
            if ((c & 0xE0) == 0xC0) expected_len = 2;
            else if ((c & 0xF0) == 0xE0) expected_len = 3;
            else if ((c & 0xF8) == 0xF0) expected_len = 4;
            else return len;  // Invalid, treat as complete

            // Check if we have all the bytes
            if (i >= (size_t)expected_len) {
                return len;  // Complete
            } else {
                return len - i;  // Incomplete, return valid portion
            }
        }
        // Continuation byte (10xxxxxx) - keep looking for start byte
    }
    return len;  // No incomplete sequence found
}

// Flush any pending output (called at end of generation)
void Backend::flush_output() {
    // Flush any partial backticks as content
    if (!backtick_buffer.empty()) {
        output_buffer += backtick_buffer;
        backtick_buffer.clear();
    }
    flush_filter_buffer();
}

// Common output filtering - handles backticks, think blocks, and buffering
// Returns true to continue, false if callback requested cancellation
bool Backend::filter(const char* text, size_t len) {
    // Handle incomplete UTF-8 from previous call
    std::string combined;
    if (!utf8_incomplete.empty()) {
        dout(2) << "filter: prepending " << utf8_incomplete.length() << " incomplete UTF-8 bytes" << std::endl;
        combined = utf8_incomplete + std::string(text, len);
        utf8_incomplete.clear();
        text = combined.c_str();
        len = combined.length();
    }

    // Check for incomplete UTF-8 at the end
    size_t valid_len = validate_utf8(std::string(text, len));
    if (valid_len < len) {
        // Buffer incomplete bytes for next call
        dout(2) << "filter: buffering " << (len - valid_len) << " incomplete UTF-8 bytes at end" << std::endl;
        utf8_incomplete = std::string(text + valid_len, len - valid_len);
        len = valid_len;
        if (len == 0) return true;  // Nothing complete yet
    }

    // Debug: log high bytes (non-ASCII)
    for (size_t i = 0; i < len; i++) {
        unsigned char uc = static_cast<unsigned char>(text[i]);
        if (uc >= 0x80) {
            dout(2) << "filter: byte " << i << " = 0x" << std::hex << (int)uc << std::dec << std::endl;
        }
    }

    for (size_t i = 0; i < len; i++) {
        char c = text[i];

        // Think block detection: look for <think> and </think>
        if (c == '<' || !think_tag_buffer.empty()) {
            think_tag_buffer += c;

            // Check for complete tags
            if (think_tag_buffer == "<think>") {
                in_think_block = true;
                think_tag_buffer.clear();
                // If thinking enabled, output the tag; otherwise suppress
                if (config && config->thinking) {
                    output_buffer += "<think>";
                }
                continue;
            } else if (think_tag_buffer == "</think>") {
                in_think_block = false;
                think_tag_buffer.clear();
                // If thinking enabled, output the tag; otherwise suppress
                if (config && config->thinking) {
                    output_buffer += "</think>";
                }
                continue;
            } else if (think_tag_buffer.length() < 8) {
                // Could still be <think> or </think>, keep buffering
                // Check if it could still match
                const char* open_tag = "<think>";
                const char* close_tag = "</think>";
                bool could_match_open = (think_tag_buffer.length() <= 7 &&
                    think_tag_buffer == std::string(open_tag, think_tag_buffer.length()));
                bool could_match_close = (think_tag_buffer.length() <= 8 &&
                    think_tag_buffer == std::string(close_tag, think_tag_buffer.length()));
                if (could_match_open || could_match_close) {
                    continue;  // Keep buffering
                }
            }

            // Doesn't match - flush tag buffer as content (unless in suppressed think block)
            if (!in_think_block || (config && config->thinking)) {
                output_buffer += think_tag_buffer;
            }
            think_tag_buffer.clear();
            continue;
        }

        // If in think block and thinking disabled, suppress content
        if (in_think_block && config && !config->thinking) {
            continue;
        }

        // Track backticks for code block detection (``` toggles code block mode)
        // Suppress the ``` markers entirely from output
        if (c == '`') {
            backtick_buffer += c;
            if (backtick_buffer.length() >= 3) {
                in_code_block = !in_code_block;
                backtick_buffer.clear();
                // Skip language identifier after opening ```
                if (in_code_block) {
                    skip_to_newline = true;
                }
            }
            continue;  // Don't add backticks to output yet
        } else {
            // If we had partial backticks (1 or 2), flush them as content
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

        // Buffer regular output
        output_buffer += c;
    }

    // Flush after processing all input (stream per-token, not per-line)
    if (!output_buffer.empty()) {
        if (!flush_filter_buffer()) {
            return false;  // Cancelled
        }
    }
    return true;
}
