#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <functional>
#include <atomic>
#include <chrono>

// Forward declarations for FTXUI
namespace ftxui {
    class ScreenInteractive;
    class Loop;
    class ComponentBase;
}

// Forward declare our Color enum (defined in terminal_io.h)
enum class Color;

// Callback type for when user submits input
using InputCallback = std::function<void(const std::string&)>;

// TUIScreen - Manages split-screen terminal layout using FTXUI
// Output area at top (grows upward from input), input at bottom, status line below
class TUIScreen {
public:
    TUIScreen();
    ~TUIScreen();

    // Initialize/shutdown TUI mode
    bool init();
    void shutdown();

    // Output line types for formatting
    enum class LineType {
        USER,       // User input ("> " prefix, green, no indent)
        TOOL_CALL,  // Tool call ("* " prefix, yellow, no indent)
        TOOL_RESULT,// Tool result ("> " prefix, cyan, no indent)
        ASSISTANT,  // Model output (default color, indented)
        SYSTEM      // System messages (red/gray, no indent)
    };

    // Output area management (thread-safe)
    void write_output(const std::string& text, LineType type);
    void write_output(const char* text, size_t len, LineType type);

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

    // Check if Escape was pressed (for cancellation) - clears the flag
    bool check_escape_pressed() {
        return escape_pressed.exchange(false);
    }

    // Request a screen refresh (call after adding output)
    void request_refresh();

    // Force immediate screen update
    void flush();

    // Legacy API compatibility (for InputReader transition)
    void set_input_content(const std::string& content);
    void position_cursor_in_input(int col);

    // Queued input display (for async generation)
    void show_queued_input(const std::string& input);
    void mark_input_processing();
    void clear_queued_input_display();

    // Configuration
    int scrollback_limit = 10000;

private:
    // FTXUI components (use raw pointers to avoid include issues)
    ftxui::ScreenInteractive* screen;
    ftxui::Loop* loop;
    std::shared_ptr<ftxui::ComponentBase> main_component;
    std::shared_ptr<ftxui::ComponentBase> input_component;

    // State
    std::string input_content;

    // Output lines with type info
    struct ColoredLine {
        std::string text;
        Color color;
        LineType type;
    };
    std::vector<ColoredLine> output_lines;
    std::string current_output_line;  // Partial line for streaming
    Color current_output_color;  // Color for current streaming line (initialized in .cpp)
    LineType current_output_type;  // Type for current streaming line
    int scroll_offset{0};  // Lines scrolled back from bottom (0 = at bottom)
    std::string status_left;
    std::string status_right;
    std::atomic<bool> quit_requested{false};
    std::atomic<bool> refresh_needed{false};
    std::atomic<bool> escape_pressed{false};  // For cancellation
    std::chrono::steady_clock::time_point last_escape_time;  // For double-escape detection
    int cursor_position{0};  // FTXUI input cursor position
    bool insert_mode{true};  // Insert vs overtype mode

    // Input history
    std::vector<std::string> history;
    int history_index{-1};
    std::string saved_input;  // Save current input when browsing history

    // Queued input display (separate from output)
    struct QueuedInputDisplay {
        std::string text;
        bool is_processing;  // false=gray (queued), true=cyan (processing)
    };
    std::vector<QueuedInputDisplay> queued_inputs;

    // Input callback
    InputCallback on_input;

    // Thread safety for output
    mutable std::mutex output_mutex;

    // Build the FTXUI component tree
    std::shared_ptr<ftxui::ComponentBase> build_ui();

    // Handle input submission
    void on_input_submit();
};

// Global TUI screen instance (created by TerminalIO when in interactive mode)
extern TUIScreen* g_tui_screen;
