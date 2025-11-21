#pragma once

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <termios.h>

// Forward declaration
struct Replxx;

// Color codes for terminal output
enum class Color {
    DEFAULT,
    RED,
    YELLOW,
    GREEN,
    CYAN,
    GRAY
};

// Marker configuration for tag detection
struct OutputMarkers {
    std::vector<std::string> tool_call_start;
    std::vector<std::string> tool_call_end;
    std::vector<std::string> thinking_start;
    std::vector<std::string> thinking_end;
};

class TerminalIO {
public:
    // Public state
    bool interactive_mode;
    bool colors_enabled;

    // Output filtering configuration
    OutputMarkers markers;
    bool show_thinking;
    std::string buffered_tool_call;     // Tool call content if detected
    std::string buffered_thinking;      // Thinking content (if show_thinking=true)
    bool last_char_was_newline;         // Track if last output char was newline

    // Constructor/Destructor
    TerminalIO();
    ~TerminalIO();

    // Initialization
    // color_override: -1 = auto-detect, 0 = force off, 1 = force on
    bool init(int color_override = -1);

    // Input/Output
    std::string read(const char* prompt);
    void write(const char* text, size_t len, Color color = Color::DEFAULT);
    void add_input(const std::string& input);

    // Response lifecycle management
    void begin_response();  // Call before starting to write a response
    void end_response();    // Call after finishing a response - flushes/consumes incomplete tags

    // Reset output filtering state between requests (deprecated - use begin_response/end_response)
    void reset();

    // Terminal control for escape key detection
    void set_raw_mode();
    void restore_terminal();
    bool check_escape_pressed();

private:
    // Input queue
    std::deque<std::string> input_queue;
    std::mutex queue_mutex;

    // Replxx instance
    Replxx* replxx;

    // Terminal state for raw mode (unused for now)
    struct termios original_term;
    bool term_raw_mode;

    // Output filtering state machine
    enum FilterState {
        NORMAL,              // Normal output
        DETECTING_TAG,       // Saw '<', buffering to detect what tag
        IN_THINKING,         // Inside thinking block
        IN_TOOL_CALL,        // Inside tool call block
        CHECKING_CLOSE       // Saw '</', checking if it closes current block
    };

    FilterState filter_state;
    bool in_tool_call;
    bool in_thinking;
    bool suppress_output;
    std::string tag_buffer;
    std::string current_tag;

    // Private methods
    std::string get_input_line(const char* prompt);
    bool is_blank(const std::string& str) const;
    const char* get_ansi_color(Color color) const;
    void write_raw(const char* text, size_t len, Color color = Color::DEFAULT);

    // Output filtering helpers
    bool matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched = nullptr) const;
    bool could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const;
};

// Global instance
extern TerminalIO tio;
