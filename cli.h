#pragma once

#include "frontend.h"
#include "session.h"
#include "backends/backend.h"
#include <string>
#include <memory>
#include <termios.h>

// Forward declaration
typedef struct Replxx Replxx;

// Run CLI mode - contains the entire interaction loop
// Returns 0 on success, non-zero on error
int run_cli(std::unique_ptr<Backend>& backend, Session& session);

// CLI class handles user interaction and tool execution
class CLI : public Frontend {
public:
    CLI();
    ~CLI();

    // Frontend interface
    int run(std::unique_ptr<Backend>& backend, Session& session) override;

    // Output functions (handle colors based on mode)
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& result);
    void show_error(const std::string& error);
    void show_cancelled();

    // Message I/O using terminal I/O system
    void send_message(const std::string& message);
    std::string receive_message(const std::string& prompt = ">");

    // Slash command handler (static for use by both CLI and interactive modes)
    static bool handle_slash_commands(const std::string& input,
                                      std::unique_ptr<Backend>& backend,
                                      Session& session);

    // Public state - direct access, no getters
    bool eof_received;
    bool generation_cancelled;
};