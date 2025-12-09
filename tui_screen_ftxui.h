#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <functional>
#include <atomic>

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
    std::vector<std::string> output_lines;
    std::string status_left;
    std::string status_right;
    std::atomic<bool> quit_requested{false};
    std::atomic<bool> refresh_needed{false};

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
