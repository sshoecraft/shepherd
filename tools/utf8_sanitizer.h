#pragma once
#include <string>
#include <cstdint>

namespace utf8_sanitizer {
    /**
     * Sanitize a string to ensure it contains only valid UTF-8 sequences.
     * Invalid UTF-8 byte sequences are replaced with replacement characters.
     * 
     * @param input The input string that may contain invalid UTF-8
     * @return A string guaranteed to contain only valid UTF-8 sequences
     */
    std::string sanitize_utf8(const std::string& input);
    
    /**
     * Check if a string contains valid UTF-8 sequences.
     * 
     * @param input The input string to validate
     * @return True if the string contains only valid UTF-8, false otherwise
     */
    bool is_valid_utf8(const std::string& input);
}