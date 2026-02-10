#pragma once

#include "frontend.h"
#include "session.h"
#include "backend.h"
#include "tools/tools.h"
#include "message.h"
#include <string>
#include <memory>
#include <deque>
#include <termios.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <pthread.h>

// Forward declaration
typedef struct Replxx Replxx;

// CLI class handles user interaction and tool execution
class CLI : public Frontend {
public:
    CLI();
    ~CLI();

    // Frontend interface
    void init(bool no_mcp = false, bool no_tools = false, bool no_rag = false) override;
    int run(Provider* cmdline_provider = nullptr) override;

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

    // Escape key handling for cancellation
    bool check_escape_key();
    void enter_generation_mode();
    void exit_generation_mode();

    // Terminal state
    bool interactive_mode = false;
    bool colors_enabled = false;
    bool at_line_start = true;  // For indentation tracking
    bool no_tools = false;   // --notools flag
    bool no_mcp = false;     // --nomcp flag (for fallback to local tools)
    bool no_rag = false;     // --norag flag

    // Unified input queue (receives from: input thread, scheduler, remote clients)
    struct QueuedInput {
        std::string text;
        bool needs_echo;
    };
    std::deque<QueuedInput> input_queue;
    std::mutex input_mutex;
    std::condition_variable input_cv;
    void add_input(const std::string& input, bool needs_echo = false);
    QueuedInput wait_for_input(int timeout_ms = -1);

    // Input reader thread
    std::thread input_thread;
    pthread_t input_thread_id = 0;
    std::atomic<bool> input_running{true};
    std::atomic<bool> ready_for_input{true};  // False during generation
    std::condition_variable ready_cv;
    std::mutex ready_mutex;
    void start_input_thread();
    void stop_input_thread();
    void input_reader_loop();
    void pause_input();   // Call before generation
    void resume_input();  // Call after generation
    static void input_signal_handler(int sig);

    // Pending scheduled prompts (injected via replxx at start of input cycle)
    std::deque<std::string> scheduled_queue;
    std::mutex scheduled_mutex;
    void add_scheduled_prompt(const std::string& prompt);

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
