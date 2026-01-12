#pragma once

#include "frontend.h"
#include "session.h"
#include "tools/tools.h"
#include "message.h"
#include <string>
#include <vector>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
#include <ncurses.h>

// Callback type for when user submits input
using InputCallback = std::function<void(const std::string&)>;


// TUI - Full ncurses TUI Frontend
// Output at top, pending queue, input box, status line at bottom
// Handles its own input and output
class TUI : public Frontend {
public:
    TUI();
    ~TUI();

    // Frontend interface
    void init(bool no_mcp = false, bool no_tools = false) override;
    int run(Provider* cmdline_provider = nullptr) override;

    // Tool management (owned by this frontend)
    Tools tools;

    // Legacy init/shutdown for backwards compatibility with main.cpp
    bool init_ncurses();
    void shutdown();

    // Output area management (thread-safe)
    // Uses CallbackEvent for color/formatting (centralized in frontend.h)
    void write_output(const std::string& text, CallbackEvent type);
    void write_output(const char* text, size_t len, CallbackEvent type);

    // Input display (called by TerminalIO)
    void set_input_content(const std::string& content);
    void position_cursor_in_input(int col);
    std::string get_input_content() const;
    void clear_input();

    // Status line
    void set_status(const std::string& left, const std::string& right);
    void update_status_bar();  // Update status with provider/token info

    // Run one iteration - handles resize and refresh (no input handling)
    // Returns false if quit was requested
    bool run_once();

    // Check if TUI has been quit
    bool has_quit() const { return quit_requested; }

    // Set/check escape pressed (called by TerminalIO's input handler)
    void set_escape_pressed() { escape_pressed = true; }
    bool check_escape_pressed() { return escape_pressed.exchange(false); }

    // Set quit flag
    void request_quit() { quit_requested = true; }

    // Request a screen refresh
    void request_refresh() { refresh_needed = true; }

    // Scroll output (Page Up/Down)
    void scroll_up() { /* TODO: implement scrollback */ }
    void scroll_down() { /* TODO: implement scrollback */ }

    // Queued input display (for async generation)
    void show_queued_input(const std::string& input);
    void mark_input_processing();
    void clear_queued_input_display();

    // Compatibility methods (may not be needed)
    void begin_output() {}
    void end_output() {}

    // Configuration
    int max_input_lines = 5;
    int max_pending_lines = 2;

    // Public state for run() access
    bool piped_eof = false;
    bool no_tools = false;  // --notools flag

    // Input queue for async input
    struct QueuedInput {
        std::string text;
        bool needs_echo;
    };
    std::deque<QueuedInput> user_input_queue;
    std::condition_variable input_cv;

    // Input management
    void add_input(const std::string& input, bool needs_echo = true);
    bool has_pending_input();
    bool wait_for_input(int timeout_ms);

    // Output helpers for tool display
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& summary, bool success);
    void show_error(const std::string& error);
    void show_cancelled();

private:
    // ncurses windows
    WINDOW* output_pad;      // Pad for scrollback output
    WINDOW* pending_win;
    WINDOW* input_win;
    WINDOW* status_win;

    // Terminal dimensions
    int term_rows;
    int term_cols;

    // Layout sizes
    int output_height;
    int pending_height;
    int input_height;
    int status_height;

    // Pad scrollback settings (pad_lines comes from config.tui_history)
    int pad_row{0};          // Current write position in pad
    int pad_view_top{0};     // Top line of viewport (for scrolling back)

    // Input state (now handled directly by TUI)
    std::string input_content;
    int input_cursor_pos{0};
    std::vector<std::string> history;
    int history_index{-1};
    std::string saved_input;
    bool in_paste_mode{false};
    std::string paste_buffer;
    std::chrono::steady_clock::time_point last_escape_time;

    // Current output state
    bool in_escape_sequence = false;
    bool at_line_start{true};  // Track if we're at start of line for indentation

    // Status
    std::string status_left;
    std::string status_right;

    // Flags
    std::atomic<bool> quit_requested{false};
    std::atomic<bool> refresh_needed{false};
    std::atomic<bool> escape_pressed{false};

    // Queued input display
    struct QueuedInputDisplay {
        std::string text;
        bool is_processing;
    };
    std::vector<QueuedInputDisplay> queued_input_display;

    // Thread safety
    mutable std::mutex output_mutex;
    mutable std::mutex input_mutex;

    // Layout and window management
    void calculate_layout();
    void create_windows();
    void destroy_windows();

    // Drawing
    void draw_output();
    void draw_pending();
    void draw_input_box();
    void draw_status();

    // Helpers
    int count_input_lines() const;
    void enable_bracketed_paste();
    void disable_bracketed_paste();

    // Input handling
    void handle_key(int ch);
    void submit_input();
    void update_input_display();
    void process_escape_sequence();

    // Output callback for backend
    bool output_callback(CallbackEvent type, const std::string& content,
                        const std::string& tool_name, const std::string& tool_call_id);
};

// Global TUI instance
extern TUI* g_tui;
