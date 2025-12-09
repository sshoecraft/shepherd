#pragma once

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <termios.h>

// Forward declarations
struct Replxx;
class TUIScreen;

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
    bool tui_mode;                      // True if using TUI screen layout

    // Output filtering configuration
    OutputMarkers markers;
    bool show_thinking;
    std::string buffered_tool_call;     // Tool call content if detected
    std::string buffered_thinking;      // Thinking content (if show_thinking=true)
    bool last_char_was_newline;         // Track if last output char was newline
    int json_brace_depth;               // Track brace depth for JSON tool calls

    // Constructor/Destructor
    TerminalIO();
    ~TerminalIO();

    // Initialization
    // color_override: -1 = auto-detect, 0 = force off, 1 = force on
    // tui_override: -1 = auto (on if interactive), 0 = force off, 1 = force on
    bool init(int color_override = -1, int tui_override = -1);

    // Input/Output
    std::string read(const char* prompt);
    std::pair<std::string, bool> read_with_echo_flag(const char* prompt);  // Returns (input, needs_echo)
    void write(const char* text, size_t len, Color color = Color::DEFAULT);
    void add_input(const std::string& input, bool needs_echo = true);

    // Queue management for producer-consumer model
    void notify_input();                    // Signal that input is available
    bool wait_for_input(int timeout_ms);    // Wait for input, returns true if available
    bool has_pending_input();               // Check if queue has pending items
    size_t queue_size();                    // Get current queue depth
    void clear_input_queue();               // Clear all pending input (used on Escape cancel)

    // Response lifecycle management
    void begin_response();  // Call before starting to write a response
    void end_response();    // Call after finishing a response - flushes/consumes incomplete tags

    // Reset output filtering state between requests (deprecated - use begin_response/end_response)
    void reset();

    // Terminal control for escape key detection
    void set_raw_mode();
    void restore_terminal();
    bool check_escape_pressed();

    // TUI-specific methods
    void set_status(const std::string& left, const std::string& right);
    void echo_user_input(const std::string& input);  // Show user input in output area
    TUIScreen* get_tui_screen() { return tui_screen; }
    Replxx* get_replxx() { return replxx; }  // For InputReader to share

    // Generation state (for queued input visual feedback)
    std::atomic<bool> is_generating{false};

private:
    // TUI screen (owned, nullptr if not in TUI mode)
    TUIScreen* tui_screen;
    // Input queue item - input string plus whether it needs to be echoed
    struct QueuedInput {
        std::string text;
        bool needs_echo;  // false if replxx already displayed it
    };
    // Input queue with condition variable for producer-consumer
    std::deque<QueuedInput> input_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    // Replxx instance (may be nullptr if InputReader owns it)
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
    bool in_code_block;        // Inside ``` code block - don't capture tool calls
    bool suppress_output;
    std::string tag_buffer;
    std::string current_tag;
    std::string backtick_buffer;  // Track consecutive backticks

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
