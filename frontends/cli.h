#pragma once

#include "frontend.h"
#include "session.h"
#include "backend.h"
#include "tools/tools.h"
#include "message.h"
#include <string>
#include <memory>
#include <queue>
#include <deque>
#include <termios.h>
#include <vector>

// Forward declaration
typedef struct Replxx Replxx;

// Pending tool call structure
struct PendingToolCall {
    std::string name;
    std::string args;
    std::string tool_call_id;
};

// CLI class handles user interaction and tool execution
class CLI : public Frontend {
public:
    CLI();
    ~CLI();

    // Frontend interface
    void init(bool no_mcp = false, bool no_tools = false) override;
    int run() override;

    // Tool management
    Tools tools;

    // Output functions (handle colors based on mode)
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& summary, bool success);
    void show_error(const std::string& error);
    void show_cancelled();

    // Message I/O
    void send_message(const std::string& message);
    std::string read_input(const std::string& prompt = "> ");  // Blocking input

    // Public state
    bool eof_received = false;
    bool generation_cancelled = false;

    // Terminal state
    bool interactive_mode = false;
    bool colors_enabled = false;
    bool at_line_start = true;  // For indentation tracking

    // Pending tool calls (filled by callback, processed by main loop)
    std::queue<PendingToolCall> pending_tool_calls;

    // Piped input support
    std::deque<std::string> piped_input_queue;
    bool piped_eof = false;
    void add_input(const std::string& input, bool needs_echo = false);

    // Output helpers
    void write_colored(const std::string& text, CallbackEvent type);
    void write_raw(const std::string& text);
    static const char* ansi_color(CallbackEvent type);

    // Replxx access for main loop
    Replxx* get_replxx() { return replxx; }

private:
    // Replxx for line editing
    Replxx* replxx = nullptr;

    // History
    std::vector<std::string> history;
    std::string history_file;

    // Terminal state for raw mode
    struct termios original_term;
    bool term_raw_mode = false;

    // Load/save history
    void load_history();
    void save_history();
};
