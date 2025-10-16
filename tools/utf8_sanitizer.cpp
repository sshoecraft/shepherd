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
            num_bytes = 2;
        } else if ((byte & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
        } else if ((byte & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
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
            if ((bytes[i + 1] & 0xC0) == 0x80) {
                num_bytes = 2;
            } else {
                // Invalid continuation byte
                result += '\xEF'; // Replacement character U+FFFD
                result += '\xBF';
                result += '?';
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
                result += '?';
                i++;
                continue;
            }
        } else if ((byte & 0xF8) == 0xF0 && i + 3 < length) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if ((bytes[i + 1] & 0xC0) == 0x80 && (bytes[i + 2] & 0xC0) == 0x80 && (bytes[i + 3] & 0xC0) == 0x80) {
                num_bytes = 4;
            } else {
                // Invalid continuation bytes
                result += '\xEF';
                result += '\xBF';
                result += '?';
                i++;
                continue;
            }
        } else {
            // Invalid UTF-8 start byte or incomplete sequence
            result += '\xEF'; // Replacement character U+FFFD
            result += '\xBF';
            result += '?';
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

} // namespace utf8_sanitizer