#include "utf8_sanitizer.h"
#include <stdexcept>
#include <cstring>

namespace utf8_sanitizer {

bool is_valid_utf8(const std::string& input) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(input.c_str());
    size_t length = input.length();
    
    for (size_t i = 0; i < length;) {
        unsigned char byte = bytes[i];
        
        // Determine number of bytes in this UTF-8 sequence
        int num_bytes;
        if ((byte & 0x80) == 0) {
            // 1-byte sequence (0xxxxxxx)
            num_bytes = 1;
        } else if ((byte & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            // Reject overlong encodings: 0xC0 and 0xC1 are invalid
            if (byte == 0xC0 || byte == 0xC1) {
                return false;
            }
            num_bytes = 2;
        } else if ((byte & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            num_bytes = 3;
        } else if ((byte & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            // Reject invalid start bytes: 0xF5-0xFF are invalid
            if (byte >= 0xF5) {
                return false;
            }
            num_bytes = 4;
        } else {
            // Invalid UTF-8 start byte
            return false;
        }
        
        // Check that we have enough bytes
        if (i + num_bytes > length) {
            return false;
        }
        
        // Validate continuation bytes
        for (int j = 1; j < num_bytes; ++j) {
            if ((bytes[i + j] & 0xC0) != 0x80) {
                return false;
            }
        }
        
        i += num_bytes;
    }
    
    return true;
}

std::string sanitize_utf8(const std::string& input) {
    // If input is already valid UTF-8, return as-is
    if (is_valid_utf8(input)) {
        return input;
    }
    
    // Process the string to replace invalid UTF-8 sequences
    std::string result;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(input.c_str());
    size_t length = input.length();
    
    for (size_t i = 0; i < length;) {
        unsigned char byte = bytes[i];
        
        // Check for valid UTF-8 start byte
        int num_bytes;
        if ((byte & 0x80) == 0) {
            // 1-byte sequence (0xxxxxxx)
            num_bytes = 1;
        } else if ((byte & 0xE0) == 0xC0 && i + 1 < length) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            // Reject overlong encodings: 0xC0 and 0xC1 are invalid
            if (byte == 0xC0 || byte == 0xC1) {
                result += '\xEF'; // Replacement character U+FFFD
                result += '\xBF';
                result += '\xBD';
                i++;
                continue;
            }
            if ((bytes[i + 1] & 0xC0) == 0x80) {
                num_bytes = 2;
            } else {
                // Invalid continuation byte
                result += '\xEF'; // Replacement character U+FFFD
                result += '\xBF';
                result += '\xBD';
                i++;
                continue;
            }
        } else if ((byte & 0xF0) == 0xE0 && i + 2 < length) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if ((bytes[i + 1] & 0xC0) == 0x80 && (bytes[i + 2] & 0xC0) == 0x80) {
                num_bytes = 3;
            } else {
                // Invalid continuation bytes
                result += '\xEF';
                result += '\xBF';
                result += '\xBD';
                i++;
                continue;
            }
        } else if ((byte & 0xF8) == 0xF0 && i + 3 < length) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            // Reject invalid start bytes: 0xF5-0xFF are invalid
            if (byte >= 0xF5) {
                result += '\xEF'; // Replacement character U+FFFD
                result += '\xBF';
                result += '\xBD';
                i++;
                continue;
            }
            if ((bytes[i + 1] & 0xC0) == 0x80 && (bytes[i + 2] & 0xC0) == 0x80 && (bytes[i + 3] & 0xC0) == 0x80) {
                num_bytes = 4;
            } else {
                // Invalid continuation bytes
                result += '\xEF';
                result += '\xBF';
                result += '\xBD';
                i++;
                continue;
            }
        } else {
            // Invalid UTF-8 start byte or incomplete sequence
            result += '\xEF'; // Replacement character U+FFFD
            result += '\xBF';
            result += '\xBD';
            i++;
            continue;
        }
        
        // Valid sequence, append it
        for (int j = 0; j < num_bytes; ++j) {
            result += bytes[i + j];
        }
        
        i += num_bytes;
    }
    
    return result;
}

std::string strip_control_characters(const std::string& input) {
    std::string result;
    result.reserve(input.length());

    for (size_t i = 0; i < input.length(); i++) {
        unsigned char ch = static_cast<unsigned char>(input[i]);

        // Keep printable ASCII, newlines, tabs, and valid UTF-8 continuation bytes
        if (ch >= 32 && ch < 127) {
            // Normal printable ASCII
            result += ch;
        } else if (ch == '\n' || ch == '\t' || ch == '\r') {
            // Allow newline, tab, carriage return
            result += ch;
        } else if (ch >= 128) {
            // UTF-8 multi-byte sequences - keep them
            result += ch;
        }
        // Skip control characters (0-31 except \n, \t, \r) and DEL (127)
    }

    return result;
}

} // namespace utf8_sanitizer