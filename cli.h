#pragma once

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
class CLI {
public:
    CLI();
    ~CLI();

    // Output functions (handle colors based on mode)
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& result);
    void show_error(const std::string& error);
    void show_cancelled();

    // Public state - direct access, no getters
    bool eof_received;
    bool generation_cancelled;
};