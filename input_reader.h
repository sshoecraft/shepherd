#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <functional>

// Forward declaration
struct Replxx;

// InputReader - Dedicated thread for reading user input
// Reads from replxx (interactive) or stdin (piped) and queues input
class InputReader {
public:
    // Callback type for when input is received
    using InputCallback = std::function<void(const std::string&)>;

    InputReader();
    ~InputReader();

    // Initialize the reader
    // interactive: true for replxx, false for stdin
    // colors_enabled: whether to use colored prompt
    // callback: function to call when input is received (typically tio.add_input)
    bool init(bool interactive, bool colors_enabled, InputCallback callback);

    // Start the input reading thread
    void start();

    // Stop the input reading thread
    void stop();

    // Check if reader is running
    bool is_running() const { return running; }

    // History management (for interactive mode)
    void history_add(const std::string& line);
    void history_load(const std::string& path);
    void history_save(const std::string& path);

    // Replxx instance (public for TerminalIO output)
    Replxx* replxx;

private:
    std::thread reader_thread;
    std::atomic<bool> running;
    std::atomic<bool> should_stop;
    bool interactive_mode;
    bool colors_enabled;
    InputCallback on_input;

    void reader_loop();
    std::string read_line_interactive();
    std::string read_line_piped();
    bool is_blank(const std::string& str) const;
};
