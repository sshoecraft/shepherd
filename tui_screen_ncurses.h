#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <functional>
#include <atomic>
#include <ncurses.h>

// Forward declare our Color enum (defined in terminal_io.h)
enum class Color;

// Callback type for when user submits input
using InputCallback = std::function<void(const std::string&)>;

// TUIScreen - Manages split-screen terminal layout using ncurses
// Output area at top, input at bottom, status line below
class TUIScreen {
public:
    TUIScreen();
    ~TUIScreen();

    // Initialize/shutdown TUI mode
    bool init();
    void shutdown();

    // Output area management (thread-safe)
    void write_output(const std::string& text, Color color);
    void write_output(const char* text, size_t len, Color color);

    // Input handling
    void set_input_callback(InputCallback callback);
    std::string get_input_content() const;
    void clear_input();

    // Status line
    void set_status(const std::string& left, const std::string& right);

    // Run one iteration of the event loop (non-blocking)
    // Returns false if user requested quit
    bool run_once();

    // Check if TUI has been quit
    bool has_quit() const { return quit_requested; }

    // Request a screen refresh (call after adding output)
    void request_refresh();

    // Force immediate screen update (flushes pending output)
    void flush();

    // Legacy API compatibility
    void set_input_content(const std::string& content);
    void position_cursor_in_input(int col);

    // Queued input display (for async generation)
    // Shows submitted input in gray, turns cyan when processing starts
    void show_queued_input(const std::string& input);
    void mark_input_processing();  // Changes first queued input from gray to cyan
    void clear_queued_input_display();  // Clear all queued display items

    // Configuration
    int scrollback_limit = 10000;
    int max_input_lines = 10;

private:
    // ncurses windows
    WINDOW* output_win;
    WINDOW* pending_win;  // Pending/queued input display
    WINDOW* input_win;
    WINDOW* status_win;

    // Terminal dimensions
    int term_rows;
    int term_cols;

    // Layout
    int output_height;
    int pending_height;   // Current height of pending box (0 when empty, up to max_pending_lines)
    int input_height;
    int status_height;
    const int max_pending_lines = 5;  // Max lines in pending box before scrolling

    // State
    std::string input_content;
    int input_cursor_pos;
    std::vector<std::string> output_lines;
    std::string current_output_line;  // Partial line being accumulated (for streaming)
    bool in_escape_sequence{false};   // For stripping ANSI escapes across chunks
    int scroll_offset{0};             // Lines scrolled back from bottom (0 = at bottom)

    // Queued input visual feedback (Step 6 from PLAN.md)
    struct QueuedInputDisplay {
        std::string text;
        bool is_processing;  // false=gray (queued), true=cyan (processing)
    };
    std::vector<QueuedInputDisplay> queued_input_display;

    std::string status_left;
    std::string status_right;
    std::atomic<bool> quit_requested{false};
    std::atomic<bool> refresh_needed{false};

    // Input history
    std::vector<std::string> history;
    int history_index;  // -1 means current input, 0+ means history entry
    std::string saved_input;  // Saves current input when browsing history

    // Bracketed paste state
    bool in_paste_mode;
    std::string paste_buffer;

    // Input callback
    InputCallback on_input;

    // Thread safety for output
    mutable std::mutex output_mutex;
    mutable std::mutex input_mutex;

    // Internal methods
    void calculate_layout();
    void create_windows();
    void destroy_windows();
    void draw_output();
    void draw_pending();
    void draw_input();
    void draw_input_box();
    void draw_status();
    void handle_key(int ch);
    void handle_paste_char(int ch);
    void submit_input();
    int count_input_lines() const;
    int get_color_pair(Color color) const;
    void enable_bracketed_paste();
    void disable_bracketed_paste();
};

// Global TUI screen instance
extern TUIScreen* g_tui_screen;
