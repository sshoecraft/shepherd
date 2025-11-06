#pragma once

#include "session.h"
#include "backends/backend.h"
#include <string>
#include <memory>
#include <termios.h>

// Run CLI mode - contains the entire interaction loop
// Returns 0 on success, non-zero on error
int run_cli(std::unique_ptr<Backend>& backend, Session& session);

// CLI class handles all terminal I/O and user interaction
class CLI {
public:
    CLI();
    ~CLI();

    // Initialize CLI - detects interactive vs piped mode
    void initialize();

    // Get input from user (handles both interactive and piped)
    std::string get_input_line();

    // Terminal control
    void set_terminal_raw();
    void restore_terminal();
    bool check_escape_pressed();

    // Output functions (handle colors based on mode)
    void show_prompt();
    void show_user_message(const std::string& msg);
    void show_assistant_message(const std::string& msg);
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& result);
    void show_error(const std::string& error);
    void show_cancelled();

    // Public state - direct access, no getters
    bool interactive_mode;
    bool eof_received;
    bool generation_cancelled;
    bool term_raw_mode;
    struct termios original_term;

private:
    // Helper to clean bracketed paste sequences
    std::string strip_bracketed_paste_sequences(const std::string& input);

    // Terminal device handle for prompt output (bypasses mpirun buffering)
    FILE* tty_handle;
};