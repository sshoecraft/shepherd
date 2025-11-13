#include "sse_parser.h"
#include "logger.h"
#include <sstream>

SSEParser::SSEParser() {
    LOG_DEBUG("SSEParser created");
}

bool SSEParser::process_chunk(const std::string& chunk, EventCallback callback) {
    if (chunk.empty()) {
        return true;
    }

    // Append chunk to buffer
    buffer_ += chunk;

    // Process complete lines
    size_t pos = 0;
    size_t newline_pos;

    while ((newline_pos = buffer_.find('\n', pos)) != std::string::npos) {
        // Extract line (handle both \n and \r\n)
        size_t line_end = newline_pos;
        if (line_end > 0 && buffer_[line_end - 1] == '\r') {
            line_end--;
        }

        std::string line = buffer_.substr(pos, line_end - pos);

        // Process the line
        if (!process_line(line, callback)) {
            return false; // Callback requested stop
        }

        pos = newline_pos + 1;
    }

    // Keep unprocessed data in buffer
    if (pos < buffer_.length()) {
        buffer_ = buffer_.substr(pos);
    } else {
        buffer_.clear();
    }

    return true;
}

bool SSEParser::process_line(const std::string& line, EventCallback callback) {
    // Empty line dispatches the event
    if (line.empty()) {
        if (!data_lines_.empty() || !event_type_.empty()) {
            return dispatch_event(callback);
        }
        return true;
    }

    // Comment lines start with :
    if (!line.empty() && line[0] == ':') {
        LOG_DEBUG("SSE comment: " + line);
        return true;
    }

    // Parse field:value format
    size_t colon_pos = line.find(':');
    std::string field;
    std::string value;

    if (colon_pos != std::string::npos) {
        field = line.substr(0, colon_pos);
        value = line.substr(colon_pos + 1);

        // Remove leading space from value (SSE spec)
        if (!value.empty() && value[0] == ' ') {
            value = value.substr(1);
        }
    } else {
        // Line with no colon is treated as field with empty value
        field = line;
    }

    // Process field
    if (field == "event") {
        event_type_ = value;
    } else if (field == "data") {
        // Accumulate data lines
        data_lines_.push_back(value);
    } else if (field == "id") {
        event_id_ = value;
    } else if (field == "retry") {
        // Ignore retry field (used for reconnection)
        LOG_DEBUG("SSE retry field ignored: " + value);
    } else {
        LOG_DEBUG("Unknown SSE field: " + field);
    }

    return true;
}

bool SSEParser::dispatch_event(EventCallback callback) {
    // Join data lines with newlines (SSE spec)
    event_data_.clear();
    for (size_t i = 0; i < data_lines_.size(); ++i) {
        if (i > 0) {
            event_data_ += "\n";
        }
        event_data_ += data_lines_[i];
    }

    // Only dispatch if we have data
    if (!event_data_.empty() || !event_type_.empty()) {
        LOG_DEBUG("SSE event - type: '" + event_type_ +
                  "', data length: " + std::to_string(event_data_.length()) +
                  ", id: '" + event_id_ + "'");

        bool continue_parsing = callback(event_type_, event_data_, event_id_);

        // Clear event state
        event_type_.clear();
        event_data_.clear();
        event_id_.clear();
        data_lines_.clear();

        return continue_parsing;
    }

    // Clear state even if no dispatch
    event_type_.clear();
    data_lines_.clear();
    event_id_.clear();

    return true;
}

void SSEParser::reset() {
    buffer_.clear();
    event_type_.clear();
    event_data_.clear();
    event_id_.clear();
    data_lines_.clear();
    LOG_DEBUG("SSEParser reset");
}