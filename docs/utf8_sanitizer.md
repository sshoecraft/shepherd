# UTF-8 Sanitizer Module Documentation

## Overview

The UTF-8 Sanitizer module (`tools/utf8_sanitizer.cpp` / `tools/utf8_sanitizer.h`) provides utilities for validating and sanitizing UTF-8 encoded strings. This is critical for handling arbitrary binary data that may be returned from tool execution or received from external sources before sending to JSON APIs.

## Functions

### `bool is_valid_utf8(const std::string& input)`

Validates whether a string contains only valid UTF-8 sequences.

**Returns**: `true` if all bytes form valid UTF-8 sequences, `false` otherwise.

**Implementation**:
- Checks each byte sequence for proper UTF-8 encoding
- Validates start bytes (1-byte: 0xxxxxxx, 2-byte: 110xxxxx, 3-byte: 1110xxxx, 4-byte: 11110xxx)
- Verifies continuation bytes match pattern 10xxxxxx
- Ensures complete sequences (no truncated multi-byte chars)

### `std::string sanitize_utf8(const std::string& input)`

Converts a potentially invalid UTF-8 string to valid UTF-8 by replacing invalid sequences.

**Returns**: A string guaranteed to contain only valid UTF-8 sequences.

**Implementation**:
- Fast path: Returns input unchanged if already valid UTF-8
- Slow path: Scans byte-by-byte, validating each UTF-8 sequence
- Invalid bytes replaced with Unicode replacement character U+FFFD (encoded as `0xEF 0xBF 0xBD`)
- Handles all sequence lengths (1-4 bytes)

**Replacement Strategy**:
- Invalid start bytes → replacement character
- Incomplete sequences → replacement character
- Invalid continuation bytes → replacement character

### `std::string strip_control_characters(const std::string& input)`

Removes control characters from input while preserving valid UTF-8 and specific whitespace.

**Preserved Characters**:
- Printable ASCII (32-126)
- Newline (`\n`), tab (`\t`), carriage return (`\r`)
- All UTF-8 multi-byte sequences (bytes ≥ 128)

**Removed Characters**:
- Control characters (0-31 except `\n`, `\t`, `\r`)
- DEL character (127)

**Use Case**: Sanitizing user input from terminals to remove escape sequences, backspace, and other control codes that could interfere with processing.

## UTF-8 Encoding Reference

| Character Range | Byte Pattern | Length |
|----------------|--------------|--------|
| U+0000 - U+007F | 0xxxxxxx | 1 byte |
| U+0080 - U+07FF | 110xxxxx 10xxxxxx | 2 bytes |
| U+0800 - U+FFFF | 1110xxxx 10xxxxxx 10xxxxxx | 3 bytes |
| U+10000 - U+10FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx | 4 bytes |

**Continuation Byte**: 10xxxxxx (0x80 - 0xBF)

**Replacement Character**: U+FFFD = `0xEF 0xBF 0xBD` (� in display)

## Usage Examples

### Binary Data Sanitization
```cpp
// Tool returns binary JPEG data
std::string jpeg_data = read_file("image.jpg");  // Contains 0xFF 0xD8...

// Sanitize for JSON transmission
std::string safe_data = utf8_sanitizer::sanitize_utf8(jpeg_data);

// Now safe to use in nlohmann::json
json j;
j["content"] = safe_data;  // Won't throw json::type_error
```

### User Input Filtering
```cpp
// User input may contain terminal escape sequences
std::string user_input = read_terminal();  // May contain \033[31m, backspace, etc.

// Strip control characters
user_input = utf8_sanitizer::strip_control_characters(user_input);

// Clean input ready for processing
```

### Validation Check
```cpp
std::string data = get_external_data();

if (!utf8_sanitizer::is_valid_utf8(data)) {
    LOG_WARN("Received invalid UTF-8, sanitizing...");
    data = utf8_sanitizer::sanitize_utf8(data);
}
```

## Integration Points

The sanitizer is used in several critical paths:

1. **CLI Tool Results** (`cli.cpp:831`): All tool execution results sanitized before JSON encoding
2. **OpenAI Streaming** (`backends/openai.cpp:938`): SSE data sanitized before JSON parsing (Azure OpenAI can return invalid UTF-8)
3. **User Input** (`cli.cpp:622`): Control character stripping for terminal input
4. **Server API** (`server/api_server.cpp`): Response content sanitization before transmission

## Recent Bug Fixes

### Replacement Character Encoding Bug (Current)

**Problem 1**: The sanitizer was creating invalid UTF-8 when replacing bad sequences. The replacement character was incorrectly encoded as:
```cpp
result += '\xEF';
result += '\xBF';
result += '?';     // BUG: '?' is 0x3F, not 0xBD
```

This created the byte sequence `0xEF 0xBF 0x3F`, which is **invalid UTF-8** (not a valid 3-byte sequence). This caused secondary errors:
```
[json.exception.type_error.316] invalid UTF-8 byte at index 2: 0x3F
```

**Solution 1**: Fixed all four instances to correctly encode U+FFFD:
```cpp
result += '\xEF';
result += '\xBF';
result += '\xBD';  // CORRECT: proper UTF-8 encoding
```

**Problem 2**: The sanitizer was accepting invalid UTF-8 bytes:
- **Overlong encodings**: Bytes 0xC0 and 0xC1 (2-byte sequences that encode values that should be 1-byte)
- **Invalid high bytes**: Bytes 0xF5-0xFF (would encode code points beyond valid Unicode range U+10FFFF)

These bytes were passing through and causing errors like:
```
[json.exception.type_error.316] invalid UTF-8 byte at index 435: 0xC0
```

**Solution 2**: Added validation in both `is_valid_utf8()` and `sanitize_utf8()`:
```cpp
// Reject overlong encodings
if (byte == 0xC0 || byte == 0xC1) {
    // Replace or return false
}

// Reject invalid 4-byte start bytes
if (byte >= 0xF5) {
    // Replace or return false
}
```

**Impact**: The sanitizer was ironically creating or passing through invalid UTF-8 while trying to fix invalid UTF-8, causing errors when processing binary tool results like JPEG images.

## Performance Considerations

- **Fast path optimization**: Returns input unchanged if valid UTF-8 (single validation pass)
- **Reserve capacity**: Output string pre-allocated to input size for minimal reallocations
- **Byte-by-byte processing**: Necessary for correctness, cannot use string operations
- **No regex**: Direct byte manipulation for performance

## Testing Notes

Test cases should cover:
- Valid UTF-8 (ASCII, 2-byte, 3-byte, 4-byte sequences)
- Invalid start bytes (0x80-0xBF, 0xF8-0xFF)
- Truncated sequences (incomplete multi-byte chars at end)
- Invalid continuation bytes (not 10xxxxxx pattern)
- Binary data (JPEG, PNG, PDF magic bytes)
- Mixed valid and invalid sequences
- Empty strings

## Dependencies

- Standard C++ library (`<string>`, `<cstring>`, `<stdexcept>`, `<cstdint>`)
- No external dependencies

## Error Handling

Functions never throw exceptions:
- `is_valid_utf8()`: Returns false on invalid input
- `sanitize_utf8()`: Replaces invalid bytes, always returns valid UTF-8
- `strip_control_characters()`: Silently removes control chars

This makes the sanitizer safe to use in any context without exception handling overhead.
