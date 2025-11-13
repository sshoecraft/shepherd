#pragma once

#include <string>
#include <functional>
#include <vector>

/// @brief Parser for Server-Sent Events (SSE) streams
/// Handles buffering incomplete lines and extracting events from SSE formatted data
class SSEParser {
public:
    /// @brief Callback invoked when a complete SSE event is received
    /// @param event The event type (or empty for data-only events)
    /// @param data The event data
    /// @param id Optional event ID
    /// @return true to continue parsing, false to stop
    using EventCallback = std::function<bool(const std::string& event,
                                            const std::string& data,
                                            const std::string& id)>;

    SSEParser();
    ~SSEParser() = default;

    /// @brief Process a chunk of SSE data
    /// @param chunk Raw data chunk from HTTP stream
    /// @param callback Function called for each complete event
    /// @return true if parsing should continue, false if callback requested stop
    bool process_chunk(const std::string& chunk, EventCallback callback);

    /// @brief Reset parser state (clears buffers)
    void reset();

    /// @brief Check if parser has incomplete data buffered
    /// @return true if there's buffered data waiting for more input
    bool has_buffered_data() const { return !buffer_.empty(); }

private:
    /// @brief Process a complete line from the SSE stream
    /// @param line The line to process (without newline)
    /// @param callback Event callback
    /// @return true to continue, false to stop
    bool process_line(const std::string& line, EventCallback callback);

    /// @brief Dispatch accumulated event to callback
    /// @param callback Event callback
    /// @return true to continue, false to stop
    bool dispatch_event(EventCallback callback);

    // Parser state
    std::string buffer_;           ///< Buffer for incomplete lines
    std::string event_type_;       ///< Current event type
    std::string event_data_;       ///< Accumulated event data
    std::string event_id_;         ///< Current event ID
    std::vector<std::string> data_lines_; ///< Multiple data lines for one event
};